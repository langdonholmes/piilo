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

PIILO uses poetry to manage dependencies. To install PIILO, follow these steps:
1. Clone the repository
2. Run `poetry install` in the root directory
3. Run `poetry run streamlit run app.py` to start the app

Once we have a package, users will be able to...
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