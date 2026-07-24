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

__all__ = [
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
