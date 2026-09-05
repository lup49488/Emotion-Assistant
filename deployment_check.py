"""Run repeatable pre-deployment checks without exposing configuration secrets."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from env_loader import load_project_env_if_enabled


BASE_DIR = Path(__file__).resolve().parent
load_project_env_if_enabled(BASE_DIR)
REQUIRED_PROJECT_FILES = (
    "Web_GUI.py", "api_server.py", "config.py", "requirements.txt", "sqlite_store.py",
    "Dockerfile", "Dockerfile.frontend", "docker-compose.yml", "docker/nginx.conf",
)
REQUIRED_MODULES = ("fastapi", "gradio", "numpy", "openai", "pytest")
FRONTEND_DIR = BASE_DIR / "frontend"
TUNNEL_WARNING_PATTERNS = (
    "timeout: no recent network activity",
    "failed to accept incoming stream",
    "failed to dial a quic connection",
    "no more connections active",
)


class CheckReporter:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"[FAIL] {message}")


def check_python(reporter: CheckReporter) -> None:
    if sys.version_info < (3, 10):
        reporter.fail(f"Python {sys.version.split()[0]} is unsupported; Python 3.10+ is required.")
    else:
        reporter.ok(f"Python {sys.version.split()[0]}")


def check_project_files(reporter: CheckReporter) -> None:
    missing = [name for name in REQUIRED_PROJECT_FILES if not (BASE_DIR / name).is_file()]
    if missing:
        reporter.fail(f"Missing required project file(s): {', '.join(missing)}")
    else:
        reporter.ok("Required project files are present")


def _normalized(text: str) -> str:
    """Collapse whitespace so a reformatted config file does not fail the checks."""
    return re.sub(r"\s+", " ", text)


def check_docker_configuration(reporter: CheckReporter) -> None:
    frontend_dockerfile = _normalized((BASE_DIR / "Dockerfile.frontend").read_text(encoding="utf-8"))
    api_dockerfile = _normalized((BASE_DIR / "Dockerfile").read_text(encoding="utf-8"))
    compose_file = (BASE_DIR / "docker-compose.yml").read_text(encoding="utf-8")
    compose_flat = _normalized(compose_file)
    nginx_config = _normalized((BASE_DIR / "docker" / "nginx.conf").read_text(encoding="utf-8"))

    checks: list[tuple[bool, str, str]] = [
        (
            "VITE_API_BASE_URL=/" in frontend_dockerfile,
            "Docker frontend build uses same-origin API requests",
            "Docker frontend build must set VITE_API_BASE_URL=/ for same-origin API requests.",
        ),
        (
            "proxy_pass http://api:8000" in nginx_config and "location /api/" in nginx_config,
            "Docker Nginx proxies /api/ to the api service",
            "Docker Nginx config must proxy /api/ to the api service.",
        ),
        (
            nginx_config.count("proxy_set_header Host $host;") >= 4,
            "Docker Nginx preserves Host headers for API proxy locations",
            "Docker Nginx API proxy locations must preserve the original Host header.",
        ),
        (
            # Nginx defaults to 1 MB, which is smaller than every upload the API accepts.
            re.search(r"client_max_body_size\s+\d+m", nginx_config) is not None,
            "Docker Nginx allows uploads as large as the API accepts",
            "Docker Nginx must raise client_max_body_size above the 1 MB default.",
        ),
        (
            "proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for" not in nginx_config,
            "Docker Nginx overwrites the forwarded client address",
            "Docker Nginx must overwrite X-Forwarded-For so callers cannot forge the rate-limit address.",
        ),
        (
            "/health/ready" in compose_flat and "headers={'Host': host}" in compose_flat,
            "Docker Compose healthcheck uses the trusted ready endpoint",
            "Docker Compose healthcheck must call /health/ready with an API_TRUSTED_HOSTS-compatible Host header.",
        ),
        (
            "API_TRUSTED_HOSTS: ${API_TRUSTED_HOSTS:-localhost,127.0.0.1,[::1]}" in compose_file,
            "Docker Compose provides local-safe trusted-host defaults",
            "Docker Compose must provide local-safe API_TRUSTED_HOSTS defaults.",
        ),
        (
            "API_TRUST_PROXY_HEADERS" in compose_file,
            "Docker Compose forwards the real client address to the API",
            "Docker Compose must set API_TRUST_PROXY_HEADERS so login rate limiting sees real addresses.",
        ),
        (
            "API_ENABLE_DOCS: ${API_ENABLE_DOCS:-false}" in compose_file,
            "Docker Compose keeps the OpenAPI page closed by default",
            "Docker Compose must default API_ENABLE_DOCS to false.",
        ),
        (
            '"127.0.0.1:8080:80"' in compose_file,
            "Docker Compose web service is bound to localhost",
            "Docker Compose web service should bind to 127.0.0.1:8080 for Tunnel/reverse-proxy deployments.",
        ),
        (
            "./users:/app/users" in compose_file and "./data:/app/data" in compose_file,
            "Docker Compose persists user and data directories",
            "Docker Compose must persist user and data directories with host volumes.",
        ),
        (
            "USER app" in api_dockerfile,
            "Docker API image drops root before running the server",
            "Docker API image must run the server as a non-root user.",
        ),
        (
            "APP_UID: ${APP_UID:-1000}" in compose_file and "APP_GID: ${APP_GID:-1000}" in compose_file,
            "Docker Compose passes host UID/GID into the API image",
            "Docker Compose must pass APP_UID/APP_GID so bind mounts are writable.",
        ),
        (
            "location = /health/ready" in nginx_config and "location = /health/live" in nginx_config,
            "Docker Nginx proxies liveness and readiness endpoints",
            "Docker Nginx must proxy /health/live and /health/ready to the API.",
        ),
    ]
    for passed, ok_message, fail_message in checks:
        reporter.ok(ok_message) if passed else reporter.fail(fail_message)


def check_dependencies(reporter: CheckReporter) -> None:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        reporter.fail(f"Missing Python dependency module(s): {', '.join(missing)}")
    else:
        reporter.ok("Core Python dependencies are importable")


def check_runtime_configuration(reporter: CheckReporter) -> None:
    backend = os.getenv("STORAGE_BACKEND", "json").strip().lower()
    if backend not in {"json", "sqlite"}:
        reporter.fail("STORAGE_BACKEND must be either 'json' or 'sqlite'.")
    else:
        reporter.ok(f"Storage backend: {backend}")

    host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1").strip()
    if host in {"0.0.0.0", "::"}:
        reporter.warn("The GUI is bound publicly; protect it with Cloudflare Access or an equivalent gateway.")
    else:
        reporter.ok(f"GUI bind address: {host or '127.0.0.1'}")

    provider = os.getenv("LLM_PROVIDER", "openai_compatible").strip().lower()
    if provider not in {"", "local_hf"} and not os.getenv("LLM_API_KEY", "").strip():
        reporter.warn("No generic LLM_API_KEY is configured; provider-specific environment keys or GUI input are required.")

    if backend == "sqlite":
        database_path = Path(os.getenv("SQLITE_DATABASE_PATH", BASE_DIR / "data" / "chatbot.db"))
        if not database_path.parent.exists():
            reporter.warn(f"SQLite parent directory will be created at startup: {database_path.parent}")

    if not os.getenv("API_SESSION_SECRET", "").strip():
        reporter.warn("API_SESSION_SECRET is not configured; signed sessions will be invalidated after a restart.")
    if host in {"0.0.0.0", "::"} and os.getenv("API_COOKIE_SECURE", "false").strip().lower() != "true":
        reporter.warn("API_COOKIE_SECURE should be true when serving the application through HTTPS.")


def _run_command(command: list[str], timeout_seconds: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=BASE_DIR,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _first_trusted_host() -> str:
    trusted_hosts = os.getenv("API_TRUSTED_HOSTS", "localhost,127.0.0.1,[::1]")
    return trusted_hosts.split(",", 1)[0].strip() or "localhost"


def _request(
    url: str,
    *,
    host: str | None = None,
    method: str = "GET",
    timeout_seconds: int = 8,
    follow_redirects: bool = True,
) -> tuple[int, str]:
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
            return None

    headers = {"User-Agent": "emotion-assistant-deployment-check/1.0"}
    if host:
        headers["Host"] = host
    request = urllib.request.Request(url, headers=headers, method=method)
    opener = urllib.request.build_opener() if follow_redirects else urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            return int(response.status), response.headers.get("location", "")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.headers.get("location", "")


def check_docker_runtime(reporter: CheckReporter) -> None:
    """Check the running Compose services, if Docker is available on this host."""
    try:
        result = _run_command(["docker", "compose", "ps", "--format", "json"], timeout_seconds=20)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        reporter.fail(f"Docker Compose status check could not run: {exc}")
        return
    if result.returncode:
        reporter.fail(f"Docker Compose status check failed: {(result.stderr or result.stdout).strip()}")
        return

    try:
        parsed = json.loads(result.stdout)
        services = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        services = []
        for line in result.stdout.splitlines():
            if line.strip():
                try:
                    services.append(json.loads(line))
                except json.JSONDecodeError:
                    reporter.fail("Docker Compose status output was not valid JSON.")
                    return
    by_service = {str(item.get("Service") or item.get("Name")): item for item in services}
    api = by_service.get("api")
    web = by_service.get("web")
    if not api or not web:
        reporter.fail("Docker Compose must have running api and web services.")
        return

    api_health = str(api.get("Health") or api.get("State") or "").lower()
    api_state = str(api.get("State") or "").lower()
    web_state = str(web.get("State") or "").lower()
    if "running" not in api_state or ("healthy" not in api_health and api_health not in {"", "running"}):
        reporter.fail(f"API service is not healthy: state={api.get('State')} health={api.get('Health')}")
    else:
        reporter.ok("Docker API service is running and healthy")
    if "running" not in web_state:
        reporter.fail(f"Web service is not running: state={web.get('State')}")
    else:
        reporter.ok("Docker web service is running")


def check_local_docker_http(reporter: CheckReporter) -> None:
    """Check the loopback Nginx origin exactly as Cloudflare Tunnel should reach it."""
    host = _first_trusted_host()
    checks = (
        ("http://127.0.0.1:8080/", "HEAD", {200, 304}),
        ("http://127.0.0.1:8080/health/live", "GET", {200}),
        ("http://127.0.0.1:8080/health/ready", "GET", {200}),
        ("http://127.0.0.1:8080/api/v1/status", "GET", {200}),
    )
    for url, method, expected in checks:
        try:
            status, _ = _request(url, host=host, method=method)
        except OSError as exc:
            reporter.fail(f"Local Docker HTTP check failed for {url}: {exc}")
            continue
        if status in expected:
            reporter.ok(f"Local Docker endpoint {url} returned HTTP {status}")
        else:
            reporter.fail(f"Local Docker endpoint {url} returned HTTP {status}; expected {sorted(expected)}.")


def check_public_http(reporter: CheckReporter, public_url: str) -> None:
    """Check the public hostname without treating Cloudflare Access redirects as failures."""
    url = public_url.rstrip("/") + "/"
    try:
        status, location = _request(url, method="HEAD", follow_redirects=False)
    except OSError as exc:
        reporter.fail(f"Public HTTPS check failed for {url}: {exc}")
        return
    if status in {200, 204, 301, 302, 303, 307, 308}:
        if "cloudflareaccess.com" in location:
            reporter.ok("Public hostname reaches Cloudflare Access and returns its login redirect")
        else:
            reporter.ok(f"Public hostname returned HTTP {status}")
    elif status in {502, 503, 504}:
        reporter.fail(f"Public hostname returned HTTP {status}; check cloudflared, Tunnel ingress, and local origin.")
    else:
        reporter.warn(f"Public hostname returned unexpected HTTP {status}.")


def check_cloudflared(reporter: CheckReporter) -> None:
    try:
        status = _run_command(["systemctl", "is-active", "cloudflared"], timeout_seconds=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        reporter.warn("systemctl is unavailable; skipping cloudflared service status.")
        return
    if status.returncode:
        reporter.fail(f"cloudflared service is not active: {(status.stdout or status.stderr).strip()}")
        return
    reporter.ok("cloudflared service is active")

    try:
        logs = _run_command(["journalctl", "-u", "cloudflared", "-n", "120", "--no-pager"], timeout_seconds=20)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        reporter.warn("journalctl is unavailable; skipping cloudflared log scan.")
        return
    if logs.returncode:
        reporter.warn(f"Could not read recent cloudflared logs: {(logs.stderr or logs.stdout).strip()}")
        return
    recent_logs = (logs.stdout + "\n" + logs.stderr).lower()
    matches = [pattern for pattern in TUNNEL_WARNING_PATTERNS if pattern in recent_logs]
    if matches:
        reporter.warn(
            "Recent cloudflared logs contain Tunnel transport warnings: "
            + ", ".join(matches)
            + ". If 502 recurs, upgrade cloudflared or test HTTP/2 transport."
        )
    else:
        reporter.ok("Recent cloudflared logs do not show common Tunnel transport failures")


def check_frontend_build(reporter: CheckReporter) -> None:
    if not (FRONTEND_DIR / "package.json").is_file():
        reporter.fail("frontend/package.json is missing.")
        return
    npm_command = "npm.cmd" if os.name == "nt" else "npm"
    for script in ("lint", "build"):
        result = subprocess.run([npm_command, "run", script], cwd=FRONTEND_DIR, check=False)
        if result.returncode:
            reporter.fail(f"Frontend npm run {script} failed.")
            return
    reporter.ok("Frontend lint and production build passed")


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def smoke_check_web(reporter: CheckReporter, timeout_seconds: int = 90) -> None:
    """Launch Gradio without model warmup, request its root page, then stop it."""
    port = _free_local_port()
    environment = os.environ.copy()
    environment.update({
        "GUI_PRELOAD_MODELS": "false",
        "GRADIO_SERVER_NAME": "127.0.0.1",
        "GRADIO_SERVER_PORT": str(port),
    })
    process = subprocess.Popen(
        [sys.executable, "Web_GUI.py"],
        cwd=BASE_DIR,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    endpoint = f"http://127.0.0.1:{port}/"
    deadline = time.monotonic() + timeout_seconds
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                reporter.fail(f"Web smoke check exited early. {output[-1000:].strip()}")
                return
            try:
                with urllib.request.urlopen(endpoint, timeout=2) as response:
                    if response.status == 200:
                        reporter.ok("Web smoke check returned HTTP 200")
                        return
            except OSError:
                time.sleep(1)
        reporter.fail(f"Web smoke check timed out after {timeout_seconds} seconds.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def smoke_check_api(reporter: CheckReporter, timeout_seconds: int = 45) -> None:
    """Launch the FastAPI adapter, request its health endpoint, then stop it."""
    port = _free_local_port()
    environment = os.environ.copy()
    environment.update({
        "API_SERVER_HOST": "127.0.0.1", "API_SERVER_PORT": str(port),
        # The smoke server is intentionally loopback-only; do not inherit a
        # public hostname allowlist that would reject its local health probe.
        "API_PUBLIC_MODE": "false",
        "API_TRUSTED_HOSTS": "localhost,127.0.0.1,[::1],testserver",
    })
    process = subprocess.Popen(
        [sys.executable, "api_server.py"],
        cwd=BASE_DIR,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    endpoint = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_seconds
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                reporter.fail(f"API smoke check exited early. {output[-1000:].strip()}")
                return
            try:
                with urllib.request.urlopen(endpoint, timeout=2) as response:
                    if response.status == 200:
                        payload = json.loads(response.read().decode("utf-8"))
                        if payload.get("status") == "ok" and "components" in payload and "metrics" in payload:
                            reporter.ok("API smoke check returned a healthy structured status report")
                            return
                        reporter.fail("API smoke check returned an invalid health payload.")
                        return
            except OSError:
                time.sleep(1)
        reporter.fail(f"API smoke check timed out after {timeout_seconds} seconds.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def run_tests(reporter: CheckReporter) -> None:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=BASE_DIR, check=False)
    if result.returncode:
        reporter.fail("pytest failed.")
    else:
        reporter.ok("pytest passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-web", action="store_true", help="Launch the Gradio app and check HTTP readiness.")
    parser.add_argument("--smoke-api", action="store_true", help="Launch the FastAPI adapter and check HTTP readiness.")
    parser.add_argument("--tests", action="store_true", help="Run the full pytest suite.")
    parser.add_argument("--frontend-build", action="store_true", help="Run frontend lint and production build.")
    parser.add_argument("--skip-dependencies", action="store_true", help="Skip Python dependency discovery.")
    parser.add_argument("--docker-runtime", action="store_true", help="Check running Docker Compose services.")
    parser.add_argument("--local-docker-http", action="store_true", help="Check the local Docker Nginx origin on 127.0.0.1:8080.")
    parser.add_argument("--public-url", help="Check a public HTTPS URL; Cloudflare Access redirects count as reachable.")
    parser.add_argument("--cloudflared", action="store_true", help="Check cloudflared service state and recent Tunnel warnings.")
    args = parser.parse_args()

    reporter = CheckReporter()
    check_python(reporter)
    check_project_files(reporter)
    check_docker_configuration(reporter)
    if not args.skip_dependencies:
        check_dependencies(reporter)
    check_runtime_configuration(reporter)
    if args.tests:
        run_tests(reporter)
    if args.frontend_build:
        check_frontend_build(reporter)
    if args.smoke_web:
        smoke_check_web(reporter)
    if args.smoke_api:
        smoke_check_api(reporter)
    if args.docker_runtime:
        check_docker_runtime(reporter)
    if args.local_docker_http:
        check_local_docker_http(reporter)
    if args.public_url:
        check_public_http(reporter, args.public_url)
    if args.cloudflared:
        check_cloudflared(reporter)

    print(f"\nCompleted with {len(reporter.failures)} failure(s) and {len(reporter.warnings)} warning(s).")
    return 1 if reporter.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
