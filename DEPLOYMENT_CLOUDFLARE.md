# Cloudflare Tunnel Public Test Deployment

This project can be exposed for public testing through one public hostname.
Keep the React frontend and FastAPI API under that same hostname:

- `https://chat.example.com/` -> React static files
- `https://chat.example.com/api/...` -> FastAPI

This layout is required by the signed session and CSRF-cookie design. Do not
place the frontend at `app.example.com` and the API at `api.example.com` unless
the cookie/CSRF design is changed first.

## 1. Configure the server environment

Create or update the server-side `.env`. Keep this file private and never put
these values in `frontend/.env*`.

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=replace-with-your-server-side-key
API_SESSION_SECRET=replace-with-a-unique-random-value-at-least-32-characters
API_COOKIE_SECURE=true
API_COOKIE_SAMESITE=lax
API_CORS_ORIGINS=https://chat.example.com
API_PRELOAD_MODELS=true
API_PUBLIC_MODE=true
API_TRUSTED_HOSTS=chat.example.com
API_ENABLE_DOCS=false
# cloudflared is the only path to FastAPI in this layout, so proxy client IP
# headers can be trusted for login abuse limiting.
API_TRUST_PROXY_HEADERS=true
```

Replace `chat.example.com` with the hostname configured in Cloudflare. If you
use a different API provider, configure its provider-specific key instead.

Public mode refuses to start unless secure cookies, a fixed session secret, and
an explicit public host allowlist are configured. The API also applies strict
security response headers, a request-size limit, and failed-login rate limits.

## 2. Start the Docker stack

Docker Compose runs the FastAPI container behind the frontend Nginx container.
Nginx serves React at `/` and proxies API requests to the internal `api:8000`
service. The host only needs to expose the web container on loopback:

```bash
cd /path/to/NLP_Project
docker compose up --build -d
```

Verify locally before exposing the service:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/api/v1/status
curl -I http://127.0.0.1:8080/
docker compose ps
```

`/health` is safe for a reverse proxy or uptime monitor. It reports storage,
RAG, warmup state, and aggregate request metrics without exposing credentials,
user data, model prompts, or API keys. Every API response also includes an
`X-Request-ID` header; use that value to correlate a browser failure with the
server log.

Run the same checks used by CI before a manual deployment:

```bash
python3 deployment_check.py --skip-dependencies
```

After the Docker stack is running on the server, run the runtime health check:

```bash
python3 deployment_check.py --skip-dependencies --docker-runtime --local-docker-http --public-url https://chat.example.com/ --cloudflared
```

Replace `chat.example.com` with the production hostname. A Cloudflare Access
`302` login redirect is considered reachable; public `502`, `503`, or `504`
responses are treated as deployment failures. The local Docker checks send the
first `API_TRUSTED_HOSTS` value as the `Host` header, so public-mode trusted-host
protection does not create a false `Invalid host header` result.

## 3. Configure Cloudflare Tunnel

In Cloudflare Zero Trust, create or select a named Tunnel and add one published
application route:

| Public hostname | Path | Local service |
| --- | --- | --- |
| `chat.example.com` | `/*` | `http://127.0.0.1:8080` |

The dashboard creates the DNS record for the hostname. If you use a locally
managed tunnel, the equivalent `config.yml` is:

```yaml
tunnel: YOUR_TUNNEL_UUID
credentials-file: /home/YOUR_USER/.cloudflared/YOUR_TUNNEL_UUID.json

ingress:
  - hostname: chat.example.com
    service: http://127.0.0.1:8080
  - service: http_status:404
```

Validate the rule order before running it:

```bash
cloudflared tunnel ingress validate
cloudflared tunnel ingress rule https://chat.example.com/api/v1/status
cloudflared tunnel ingress rule https://chat.example.com/
```

For a remotely managed tunnel, install it as a service using the token copied
from the Cloudflare dashboard:

```bash
sudo cloudflared service install YOUR_TUNNEL_TOKEN
sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared
```

## 4. Public checks

1. Open `https://chat.example.com` and log in.
2. In browser developer tools, confirm that the session and CSRF cookies are
   both `Secure` and sent to the same hostname.
3. Send a chat message and confirm that `POST /api/v1/chat/stream` succeeds.
4. Check the server logs and `systemctl status cloudflared`.
   Search logs by `request_id=<X-Request-ID>` when investigating a failed chat request.

Use a named tunnel for this project. Quick Tunnels (`trycloudflare.com`) are
not suitable because they do not support the application's SSE chat stream.

## 5. Deployed revision and rollback

`.github/workflows/deploy.yml` checks the working copy out at the exact commit
CI verified, so `/opt/Emotion-Assistant` sits on a **detached HEAD**. A manual
`git pull` there does nothing; deploy through the workflow instead. Trigger it
manually with `workflow_dispatch` and a full 40-character `commit_sha`, or leave
that input empty to deploy the current `origin/main` tip.

Each deployment records where it came from:

```bash
cat /opt/Emotion-Assistant/.deployment/current_sha    # what is running now
cat /opt/Emotion-Assistant/.deployment/previous_sha   # what it replaced
```

To roll back, re-run the workflow with `commit_sha` set to the recorded
`previous_sha`, so the rollback goes through the same health checks. Only if the
workflow itself is unavailable, do it on the host:

```bash
cd /opt/Emotion-Assistant
git checkout --detach "$(cat .deployment/previous_sha)"
docker compose up --build -d
python3 deployment_check.py --skip-dependencies --docker-runtime --local-docker-http
```
