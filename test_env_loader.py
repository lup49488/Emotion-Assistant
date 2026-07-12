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
