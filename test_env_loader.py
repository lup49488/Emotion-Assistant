import os
import tempfile
import unittest
from pathlib import Path

from env_loader import load_project_env


class TestEnvLoader(unittest.TestCase):
    def test_loads_dotenv_without_overriding_existing_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            (base_dir / ".env").write_text(
                "\n".join([
                    "EXISTING_VALUE=from_env_file",
                    "PLAIN_VALUE=hello",
                    'QUOTED_VALUE="hello world"',
                    "COMMENTED_VALUE=value # comment",
                ]),
                encoding="utf-8",
            )

            old_existing = os.environ.get("EXISTING_VALUE")
            try:
                os.environ["EXISTING_VALUE"] = "from_process"
                loaded = load_project_env(base_dir)

                self.assertEqual(loaded, [base_dir / ".env"])
                self.assertEqual(os.environ["EXISTING_VALUE"], "from_process")
                self.assertEqual(os.environ["PLAIN_VALUE"], "hello")
                self.assertEqual(os.environ["QUOTED_VALUE"], "hello world")
                self.assertEqual(os.environ["COMMENTED_VALUE"], "value")
            finally:
                if old_existing is None:
                    os.environ.pop("EXISTING_VALUE", None)
                else:
                    os.environ["EXISTING_VALUE"] = old_existing
                for key in ["PLAIN_VALUE", "QUOTED_VALUE", "COMMENTED_VALUE"]:
                    os.environ.pop(key, None)

    def test_dotenv_local_overrides_dotenv_but_not_process_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            (base_dir / ".env").write_text(
                "FILE_ONLY=from_env\nSHARED_VALUE=from_env\nPROCESS_VALUE=from_env\n",
                encoding="utf-8",
            )
            (base_dir / ".env.local").write_text(
                "SHARED_VALUE=from_local\nPROCESS_VALUE=from_local\n",
                encoding="utf-8",
            )

            old_process = os.environ.get("PROCESS_VALUE")
            try:
                os.environ["PROCESS_VALUE"] = "from_process"
                loaded = load_project_env(base_dir)

                self.assertEqual(loaded, [base_dir / ".env", base_dir / ".env.local"])
                self.assertEqual(os.environ["FILE_ONLY"], "from_env")
                self.assertEqual(os.environ["SHARED_VALUE"], "from_local")
                self.assertEqual(os.environ["PROCESS_VALUE"], "from_process")
            finally:
                if old_process is None:
                    os.environ.pop("PROCESS_VALUE", None)
                else:
                    os.environ["PROCESS_VALUE"] = old_process
                for key in ["FILE_ONLY", "SHARED_VALUE"]:
                    os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()


class TestSkipGuard(unittest.TestCase):
    """Every import-time loader must honour CHATBOT_SKIP_DOTENV.

    A module that calls load_project_env() directly injects the developer's real
    .env into os.environ for the whole process; in a test session that leaks into
    every module imported afterwards and makes results depend on file order.
    """

    def test_skip_guard_prevents_loading(self):
        from env_loader import load_project_env_if_enabled

        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            (base_dir / ".env").write_text("GUARD_PROBE=leaked", encoding="utf-8")
            old = os.environ.get("CHATBOT_SKIP_DOTENV")
            try:
                os.environ["CHATBOT_SKIP_DOTENV"] = "1"
                self.assertEqual(load_project_env_if_enabled(base_dir), [])
                self.assertIsNone(os.environ.get("GUARD_PROBE"))

                os.environ["CHATBOT_SKIP_DOTENV"] = "0"
                self.assertEqual(load_project_env_if_enabled(base_dir), [base_dir / ".env"])
                self.assertEqual(os.environ["GUARD_PROBE"], "leaked")
            finally:
                os.environ.pop("GUARD_PROBE", None)
                if old is None:
                    os.environ.pop("CHATBOT_SKIP_DOTENV", None)
                else:
                    os.environ["CHATBOT_SKIP_DOTENV"] = old

    def test_no_module_bypasses_the_guard(self):
        import pathlib
        import re

        offenders = []
        for path in pathlib.Path(__file__).parent.glob("*.py"):
            if path.name in {"env_loader.py", "test_env_loader.py"}:
                continue
            text = path.read_text(encoding="utf-8")
            if re.search(r"(?<!_if_enabled)\bload_project_env\(", text):
                offenders.append(path.name)

        self.assertEqual(offenders, [])
