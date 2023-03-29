---
title: Piilo
emoji: 🏃
colorFrom: purple
colorTo: purple
sdk: streamlit
sdk_version: 1.10.0
app_file: app.py
pinned: false
license: apache-2.0
---

Currently, the best way to install PIILO is using pipenv:

1. Clone the repository
    - `git clone https://huggingface.co/spaces/langdonholmes/piilo`

2. Install dependencies from Pipfile
    - Install pipenv, if you do not have it.
        - `pip install --user pipenv`

    - Use pipenv to install from the Pipfile
        - `pipenv install`

3. Install the finetuned transformer

```
pipenv install https://huggingface.co/langdonholmes/en_student_name_detector/resolve/main/en_student_name_detector-any-py3-none-any.whl
```

4. Add PIILO to path
    - Navigate to PIILO repository on your filesystem: `cd piilo`
    - `pipenv install -e .`
    
5. Use piilo in your project
```
import piilo

texts = ['test string without identifiers', 'My name is Antonio. Email: Antonio99@yahoo.com']

# To analyze the texts. Returns list of RecognizerResult, defined by presidio_analyzer
results = [piilo.analyze(text) for text in texts]

# To analyze AND anonymize with hiding-in-plain-sight obfuscation. Returns list of texts with identifiers obfuscated.
cleaned_texts = [piilo.anonymize(text) for text in texts]
```

TODO:
Create a command line version using Typer in this same repository.