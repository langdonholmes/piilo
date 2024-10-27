# Overview

This `readme` file is intended to help developers navigate through PIILO.

# Getting Started

## Using Poetry

Version control for this package relies on [Poetry](https://python-poetry.org/). You can activate the environment and install the dependencies defined in `poetry.lock` (or `pyproject.toml` if there is no `poetry.lock`) by running:

```bash
poetry shell
poetry install
```

## Getting Required Resources

Certain files are too big to be included in this repo. These files are automatically downloaded if you are accessing the package through `pip`. If you are a developer working with this repo, you can access these files individually from the following links:

Download the `xgb` models and `vectorizers` from [this link](https://www.kaggle.com/code/devinanzelmo/piidd-efficiency-3rd-inference/input?select=xgb_final) and add them to `piilo\data`.

Download `parquet` files from [this link](https://drive.google.com/drive/folders/1Ru3bLULgt-FhqgaL90uoAV69pHOZ5el0?usp=drive_link) and add them to `piilo\data`.

# PIILO as a Package

## Package Structure

```
piilo
├── setup.py  
├── README.md
├── app.py  # used for a Streamlit GUI; not a part of the package
└── piilo
    ├── __init__.py
    ├── data 
    ├── engines 
    ├── models
    └── configs
```

## Building and Uploading PIILO as a Package

Use the packages `setuptools` and `wheels` to build wheels. Make sure to update the `VERSION` variable value in `setup.py` before you run the command to ensure that you are assigning a higher version number to the current build.

```bash
python setup.py sdist bdist_wheel
```

Running the command above from a CLI will create the folders `dist`, `build`, and `piilo.egg-info` each containing distributables, temporary files from the build process, and package metadata. 

Then, run the following command to upload to PyPI or Test PyPI:

```bash
# For testpypi
twine upload --repository testpypi dist/*
# For pypi
twine upload dist/*
```

You will need an API token from the respective repository. You can use `pip` to download and use this package.

```
# For testpypi
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ piilo
# For pypi
pip install piilo
```

## PIILO as a Package: Overview

PIILO has two different engines: an `analyzer` and an `anonymizer`. The `analyzer` is based on `presidio_analyzer` and custom code from the PIILO Kaggle competition, and its goal is to analyze the text to find and tag PII for the `anonymizer` to obfuscate. The `anonymizer` is based on `presidio_anonymizer` and custom code, and its job is to obfuscate the PII to implement hiding in plain sight (HIPS).

The main functions available in the PIILO package are defined in `main.py`. 

The `analyze` function takes in a raw string as an argument, and uses a custom analyzer object to create a list of `RecognizerResult` objects. This object specified the `entity_type` of the PII and the start and end indices of the PII within the text. Currently, the custom analyzer uses a `spacy` model (`en_core_web_sm`) and a customized model (from the Kaggle competition) to find and tag PII, then prunes the results to remove duplicate tags (i.e., cases where PII was tagged by both models). To add more custom models to this pipeline, define a custom `LocalRecognizer` object in `analyze.py`, then add the model to `CustomAnalyzer` using `registry.add_recognizer(<name_of_new_model>)`. To modify the pruning behavior, change the `prune_results` static method for `CustomAnalyzer`. Refer to the [presidio doc](https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/recognizer_result.py) for more detail.

The `anonymize` function takes in a raw string as an argument; it executes the `analyze` function defined above, then uses the resulting list of `RecognizerResult` to find PII to obfuscate based on a `SurrogateAnonymizer` object, then returns an obfuscated string. Use `get_anonymize` instead to get an `AnonymizeResponse` object instead of the obfuscated string.

Below is a code snippet that shows you how you might use the two functions `analyze` and `anonymize`.

```python
import piilo

texts = ['test string without identifiers', 'My name is Antonio. Email: Antonio99@yahoo.com']

# To analyze the texts. Returns list of RecognizerResult, defined by presidio_analyzer
results = [piilo.analyze(text) for text in texts]

# To analyze AND anonymize with hiding-in-plain-sight obfuscation. Returns list of texts with identifiers obfuscated.
cleaned_texts = [piilo.anonymize(text) for text in texts]
```

## PIILO with CLI: Overview

The `anonymize_batch` and `anonymize_batch_cli` functions defined in `main.py` are functions intended to be used with CLI. The entry point for using PIILO is defined in `setup.py` like so:

```python
entry_points={
    "console_scripts": [
        "obfuscate=piilo:anonymize_batch_cli",
    ],
}
```

Modify the arguments that the `obfuscate` command can take in by using `argparse`.

# PIILO with GUI

## Streamlit App

This repo includes a streamlit app that utilizes the PIILO package. It relies on the `CustomAnalyzer` and the `SurrogateAnonymizer` defined in `engines`. The app is accessible through the python file `app.py` at the top level. Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

## Packaging PIILO as an Executable

The Streamlit app can be packaged into an executable using different executable creators (e.g., [Nutika](https://nuitka.net/), [py2exe](https://www.py2exe.org/), [cx_freeze](https://cx-freeze.readthedocs.io/en/stable/), etc.). Executable creation has been tested using [Pyinstaller](https://pyinstaller.org/en/stable/).

To create an executable with pyinstaller, create a spec file first:

```
pyi-makespec --onefile app.py  # to create a one-file bundle
```

Then modify the spec file so that the `data` argument includes paths to all subdirectories required to run PIILO. Refer to the [pyinstaller docs](https://pyinstaller.org/en/stable/spec-files.html) for more information on this. Then, create a bundle using pyinstaller with:

```
pyinstaller options <<name>>.spec
```

This will generate an executable file in the `dist` folder.