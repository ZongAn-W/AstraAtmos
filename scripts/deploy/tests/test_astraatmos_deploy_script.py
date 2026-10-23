from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "deploy" / "astraatmos.sh"


class DeployScriptContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_declares_required_commands(self):
        for command in [
            "install",
            "deploy",
            "update",
            "restart",
            "stop",
            "status",
            "logs",
            "health",
            "backup",
            "help",
        ]:
            with self.subTest(command=command):
                self.assertRegex(self.text, rf"\b{re.escape(command)}\)")

    def test_uses_systemd_service_with_single_uvicorn_worker(self):
        self.assertIn("astraatmos.service", self.text)
        self.assertIn("systemctl", self.text)
        self.assertIn(
            "python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1",
            self.text,
        )

    def test_preserves_env_and_uses_frontend_dist(self):
        self.assertIn("ASTRAATMOS_FRONTEND_DIST", self.text)
        self.assertIn("frontend_dist", self.text)
        self.assertIn('if [[ ! -f "$ENV_FILE" ]]', self.text)
        self.assertNotIn('cp -f "$ENV_EXAMPLE" "$ENV_FILE"', self.text)

    def test_installs_expected_runtime_dependencies(self):
        self.assertIn("python3-venv", self.text)
        self.assertIn("build-essential", self.text)
        self.assertIn("NODE_MAJOR=20", self.text)
        self.assertIn("torch==2.5.1", self.text)
        self.assertIn("cu124", self.text)
        self.assertIn("npm ci", self.text)
        self.assertIn("npm run build", self.text)

    def test_system_package_installs_are_noninteractive(self):
        self.assertIn("DEBIAN_FRONTEND=noninteractive", self.text)
        self.assertIn("NEEDRESTART_MODE=a", self.text)
        self.assertIn("--force-confdef", self.text)
        self.assertIn("--force-confold", self.text)

    def test_node_install_prepares_nodesource_and_verifies_runtime(self):
        self.assertIn("/etc/apt/keyrings", self.text)
        self.assertIn("/etc/apt/sources.list.d", self.text)
        self.assertIn("verify_node_runtime", self.text)
        self.assertIn("Node.js installation did not provide", self.text)
        self.assertIn("npm is not available after Node.js installation", self.text)

    def test_backend_dependencies_use_configurable_pypi_mirror(self):
        self.assertIn("PYPI_INDEX_URL", self.text)
        self.assertIn("pypi.tuna.tsinghua.edu.cn", self.text)
        self.assertIn('install --index-url "${PYPI_INDEX_URL}" -r "${req_tmp}"', self.text)

    def test_update_uses_fast_forward_only_pull(self):
        self.assertIn("git pull --ff-only", self.text)

    def test_backup_targets_runtime_state(self):
        self.assertIn("astraatmos-backups", self.text)
        for path in [".env", "data", "models", "logs"]:
            with self.subTest(path=path):
                self.assertIn(path, self.text)


if __name__ == "__main__":
    unittest.main()
