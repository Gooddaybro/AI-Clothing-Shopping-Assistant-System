import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests.contract_support import resolve_contract_root


class ContractSupportTests(unittest.TestCase):
    def test_environment_contract_root_has_priority(self) -> None:
        with TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"OUTFIT_CONTRACT_ROOT": directory}):
                self.assertEqual(Path(directory), resolve_contract_root())


if __name__ == "__main__":
    unittest.main()
