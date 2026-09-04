from pathlib import Path
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]


class CiWorkflowTests(unittest.TestCase):
    def test_python_ci_has_repeatable_quality_and_contract_gates(self) -> None:
        workflow = (PROJECT_DIR / ".github" / "workflows" / "code-quality.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("concurrency:", workflow)
        self.assertIn("timeout-minutes:", workflow)
        self.assertIn("OUTFIT_CONTRACT_ROOT", workflow)
        self.assertIn("python -m compileall", workflow)
        self.assertIn("ruff check", workflow)
        self.assertIn("interrogate", workflow)
        self.assertIn("python -m pytest -q", workflow)


if __name__ == "__main__":
    unittest.main()
