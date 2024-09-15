# PIILO
** Personally Identifiable Information Labeling and Obfuscation **

## What is PIILO?
PIILO is an open-source automatic deidentification system for student text. PIILO includes both transformer-based and rule-based systems for labeling PII in student text. One of the core design principles behind PIILO is that obfuscation is as important as labeling PII in text deidentification systems. PIILO implements obfuscation by way of HIPS (hiding in plain sight). It uses a surrogate name generator that automatically obfuscates student names with realistic and contextually plausible surrogate names. 

## Installation
From your terminal, run:

```bash
$ pip install piilo
```

## Use PIILO with a command line interface
```bash
$ obfuscate
```

Here are the arguments you can use
```
  --dir DIR             Directory containing text files to anonymize; defaults to the current directory
  --entities [ENTITIES ...]
                        Entities to anonymize
  --language LANGUAGE   Language of the text files; currently only supports 'en'
  --file_format {csv,txt}
                        Output file format; currently supports 'csv' and 'txt'
```

## Use PIILO as a Python package
```
import piilo

texts = ['test string without identifiers', 'My name is Antonio. Email: Antonio99@yahoo.com']

# To analyze the texts. Returns list of RecognizerResult, defined by presidio_analyzer
results = [piilo.analyze(text) for text in texts]

# To analyze AND anonymize with hiding-in-plain-sight obfuscation. Returns list of texts with identifiers obfuscated.
cleaned_texts = [piilo.anonymize(text) for text in texts]
```

## Use PIILO with a graphic user interface
You can also download PIILO as an executable from [this link](https://www.linguisticanalysistools.org/piilo.html)

## Cite PIILO
You can cite PIILO by using the reference below

    @article{holmes2023piilo,
    title={PIILO: an open-source system for personally identifiable information labeling and obfuscation},
    author={Holmes, Langdon and Crossley, Scott and Sikka, Harshvardhan and Morris, Wesley},
    journal={Information and Learning Sciences},
    volume={124},
    number={9/10},
    pages={266--284},
    year={2023},
    publisher={Emerald Publishing Limited}
    }

