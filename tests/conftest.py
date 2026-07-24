from pathlib import Path

import pytest

# Resolved by path rather than by importing piilo, so that collection works even
# when the model files are absent.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "piilo"

REQUIRED_ASSETS = [
    "data/ascii_names.parquet",
    "data/third_place_first_names.parquet",
    "data/third_place_last_names.parquet",
    "models/vectorizer2_raw_final.pkl",
    "models/vectorizer2_postags_final.pkl",
]

MISSING = [rel for rel in REQUIRED_ASSETS if not (PACKAGE_ROOT / rel).exists()]

requires_assets = pytest.mark.skipif(
    bool(MISSING),
    reason=(
        "Model/data files are not bundled in the repository; see DEV_README.md "
        "for download links. Missing: " + ", ".join(MISSING)
    ),
)


@pytest.fixture(scope="session")
def piilo_module():
    """Import piilo once; the first analyze() call loads the models."""
    import piilo

    return piilo
