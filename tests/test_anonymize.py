import csv

import pytest
from conftest import requires_assets

from piilo.engines.analyzer import KaggleThirdAnalyzer

SAMPLE = (
    "Learning Reflection\n\n"
    "Written by John Williams and Samantha Morales\n\n"
    "In this course I learned many things.\n\n"
    "By John H. Williams -- (714) 328-9989 -- johnwilliams@yahoo.com"
)

# Identifiers that must not survive anonymization of SAMPLE.
LEAKABLE = ["John", "Williams", "Samantha", "Morales", "johnwilliams@yahoo.com"]


def test_declared_entities_match_emitted_labels():
    """Every label the recognizer can emit must be declared to Presidio.

    A label emitted but not declared is invisible to callers who pass an
    explicit `entities` list; a label declared but never emitted is dead config.
    """
    import piilo.engines.analyzer as analyzer_module

    source = (analyzer_module.__file__ or "").replace("\\", "/")
    with open(source, "r", encoding="utf-8") as f:
        text = f.read()

    emitted = set()
    for line in text.splitlines():
        if "create_result(" in line and '"' in line:
            label = line.split('create_result("', 1)[-1].split('"', 1)[0]
            # B-/I- prefixed labels are internal and rewritten before returning.
            if not label.startswith(("B-", "I-")):
                emitted.add(label)

    declared = set(KaggleThirdAnalyzer.ENTITIES)
    assert (
        emitted - declared == set()
    ), f"emitted but not declared: {emitted - declared}"


@requires_assets
def test_analyze_returns_results(piilo_module):
    results = piilo_module.analyze(SAMPLE)
    assert results, "analyzer returned no results for text full of PII"
    labels = {r.entity_type for r in results}
    assert "PERSON" in labels
    assert "EMAIL_ADDRESS" in labels


@requires_assets
def test_anonymize_removes_identifiers(piilo_module):
    cleaned = piilo_module.anonymize(SAMPLE).text
    for identifier in LEAKABLE:
        assert identifier not in cleaned, f"{identifier!r} survived anonymization"


@requires_assets
def test_entity_filter_does_not_weaken_anonymization(piilo_module):
    """Regression: restricting `entities` used to drop Presidio's own
    recognizers, losing multi-token name spans and leaking surnames."""
    filtered = piilo_module.anonymize(
        SAMPLE, entities=KaggleThirdAnalyzer.ENTITIES
    ).text
    for identifier in LEAKABLE:
        assert (
            identifier not in filtered
        ), f"{identifier!r} survived anonymization when entities= was passed"


@requires_assets
def test_analyze_handles_text_without_names(piilo_module):
    """Regression: an empty name set used to crash feature generation."""
    assert piilo_module.analyze("There is nothing sensitive here.") is not None


@requires_assets
def test_anonymize_batch_writes_csv(piilo_module, tmp_path):
    (tmp_path / "a.txt").write_text(SAMPLE, encoding="utf-8")

    piilo_module.anonymize_batch(str(tmp_path), file_format="csv")

    out = tmp_path / "anonymized_results.csv"
    assert out.exists()
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["file_name"] == "a.txt"
    for identifier in LEAKABLE:
        assert identifier not in rows[0]["anonymized_text"]


@requires_assets
def test_anonymize_batch_writes_txt(piilo_module, tmp_path):
    (tmp_path / "a.txt").write_text(SAMPLE, encoding="utf-8")

    piilo_module.anonymize_batch(str(tmp_path), file_format="txt")

    outputs = list(tmp_path.glob("*_anonymized.txt"))
    assert len(outputs) == 1
    assert "Williams" not in outputs[0].read_text(encoding="utf-8")


@requires_assets
def test_anonymize_batch_rejects_unknown_format(piilo_module, tmp_path):
    with pytest.raises(ValueError, match="not supported"):
        piilo_module.anonymize_batch(str(tmp_path), file_format="pdf")


@requires_assets
def test_anonymize_batch_rejects_missing_directory(piilo_module, tmp_path):
    with pytest.raises(FileNotFoundError):
        piilo_module.anonymize_batch(str(tmp_path / "nope"))
