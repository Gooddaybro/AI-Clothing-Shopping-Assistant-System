import os
from pathlib import Path


def resolve_contract_root() -> Path:
    configured = os.getenv("OUTFIT_CONTRACT_ROOT")
    if configured:
        return Path(configured)

    project_dir = Path(__file__).resolve().parents[1]
    candidates = (
        project_dir / "contracts",
        project_dir.parent / "outfit-project-contract" / "contracts",
        project_dir.parent / "Intelligent Outfit Recommendation System" / "contracts",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return candidates[0]
