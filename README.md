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

For development, try working with pipenv:

1. Clone the repository
`git clone https://huggingface.co/spaces/langdonholmes/piilo`

2. Install dependencies from Pipfile
    - Install pipenv, if you do not have it.
        - `pip install --user pipenv`

    - Use pipenv to install from the Pipfile
        - `pipenv install`

3. Install the finetuned transformer model

```
pipenv install https://huggingface.co/langdonholmes/en_student_name_detector/resolve/main/en_student_name_detector-any-py3-none-any.whl
```

4. Run streamlit app
    - `streamlit run .\app.py`

TODO:
Create a command line version using Typer in this same repository.