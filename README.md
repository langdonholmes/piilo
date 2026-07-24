<div align="center">

<img src="https://raw.githubusercontent.com/langdonholmes/piilo/main/logo.png" alt="PIILO logo" width="140">

# PIILO

**P**ersonally **I**dentifiable **I**nformation **L**abeling and **O**bfuscation

[![Tests](https://github.com/langdonholmes/piilo/actions/workflows/ci.yml/badge.svg)](https://github.com/langdonholmes/piilo/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/langdonholmes/piilo/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-ILS%202023-b31b1b.svg)](https://doi.org/10.1108/ILS-04-2023-0032)

</div>

## What is PIILO?

PIILO is an open-source deidentification system for student-generated text. Most
deidentification tools stop at *finding* personally identifiable information and
replace it with a redaction marker such as `<PERSON>`. PIILO treats obfuscation
as equally important: it replaces identifiers with realistic, contextually
plausible surrogates, an approach known as **HIPS** (hiding in plain sight).

Redaction markers advertise exactly where sensitive content used to be and make
the resulting text awkward to read or model. Surrogates keep the document
natural, so deidentified text stays usable for downstream research.

```
Written by John Williams -- (714) 328-9989 -- johnwilliams@yahoo.com
                          |
                          v
Written by Stephen Yu    -- 272-947-1857   -- harveylisa@yahoo.com
```

Detection combines spaCy's named entity recognition with a rule- and
feature-based recognizer adapted from the third-place solution to the Kaggle
[PII Detection Removal from Educational Data](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)
competition, which uses gradient-boosted trees over name lists and lexical
features rather than a transformer.

## Installation

```bash
pip install piilo
```

The name tables and XGBoost models PIILO needs (roughly 120 MB) are bundled in
the release, so `pip install` gives a fully working package with nothing else to
download.

On macOS, XGBoost also needs the OpenMP runtime: `brew install libomp`.

> **Note**
> Those model files are **not** stored in the git repository, only in the
> published package. If you install from a clone (`pip install -e .`) you must
> fetch them separately — see [DEV_README.md](https://github.com/langdonholmes/piilo/blob/main/DEV_README.md).

## Usage

### As a Python package

```python
import piilo

texts = [
    "test string without identifiers",
    "My name is Antonio. Email: Antonio99@yahoo.com",
]

# Locate PII. Returns presidio_analyzer.RecognizerResult objects.
results = [piilo.analyze(text) for text in texts]

# Locate and obfuscate PII with hiding-in-plain-sight surrogates.
cleaned = [piilo.anonymize(text).text for text in texts]
```

The models load on first use rather than at import, so the first call is slower
than those that follow.

### From the command line

`obfuscate` anonymizes every `.txt` file in a directory:

```bash
obfuscate --dir ./essays --file_format csv
```

| Option | Description |
| --- | --- |
| `--dir` | Directory containing text files to anonymize. Defaults to the current directory, with a confirmation prompt. |
| `--entities` | Restrict analysis to specific entity types. Defaults to all recognizers. |
| `--language` | Language of the text files. Currently only `en`. |
| `--file_format` | `csv` (one row per file) or `txt` (one output file per input). |

### With a graphical interface

A Streamlit app is included for interactive exploration:

```bash
pip install -e ".[app]"
streamlit run app.py
```

A packaged desktop build is also available from
[linguisticanalysistools.org](https://www.linguisticanalysistools.org/piilo.html).

## Detected entities

| Entity | Obfuscation |
| --- | --- |
| `PERSON` | Surrogate name matched on inferred gender and country of origin |
| `EMAIL_ADDRESS` | Generated address |
| `PHONE_NUMBER` | Generated number |
| `URL` | Generated URL |
| `STREET_ADDRESS` | Generated address |
| `ID_NUM` | Generated identifier |
| `DATE_TIME` | Date shifted while preserving format |
| `LOCATION` | Preserved |

Surrogates are consistent within a document: the same name is always replaced by
the same surrogate, so coreference survives deidentification.

## Development

See [DEV_README.md](https://github.com/langdonholmes/piilo/blob/main/DEV_README.md)
for the package layout, how to obtain the model files, and how to build a release.

```bash
pip install -e ".[dev]"
pytest
```

Tests that need the model files skip automatically when those files are absent.

## Citation

If you use PIILO in your research, please cite:

```bibtex
@article{holmes2023piilo,
  title={PIILO: an open-source system for personally identifiable information labeling and obfuscation},
  author={Holmes, Langdon and Crossley, Scott and Sikka, Harshvardhan and Morris, Wesley},
  journal={Information and Learning Sciences},
  volume={124},
  number={9/10},
  pages={266--284},
  year={2023},
  publisher={Emerald Publishing Limited},
  doi={10.1108/ILS-04-2023-0032}
}
```

## License

Apache License 2.0. See [LICENSE](https://github.com/langdonholmes/piilo/blob/main/LICENSE).
