from importlib.metadata import PackageNotFoundError, version

import spacy

from .main import (
    analyze,
    anonymize,
    anonymize_batch,
    anonymize_batch_cli,
    get_analyzer,
    get_anonymize,
    get_anonymizer,
)

# The import package is "piilo"; the PyPI distribution is "piilo-anonymizer".
try:
    __version__ = version("piilo-anonymizer")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "analyze",
    "anonymize",
    "anonymize_batch",
    "anonymize_batch_cli",
    "get_analyzer",
    "get_anonymize",
    "get_anonymizer",
]

# en_core_web_sm as a dependency requires external link
# which does not play nice with CLI
try:
    import en_core_web_sm
except ModuleNotFoundError:
    spacy.cli.download("en_core_web_sm")
