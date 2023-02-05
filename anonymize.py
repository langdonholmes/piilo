from spacy_recognizer import CustomSpacyRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer.entities import OperatorConfig
import pandas as pd
from json import JSONEncoder
import json
import warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

def prepare_analyzer(configuration):
    """Return AnalyzerEngine."""

    spacy_recognizer = CustomSpacyRecognizer()

    print('Hallej')

    # Create NLP engine based on configuration
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    # add rule-based recognizers
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    registry.add_recognizer(spacy_recognizer)

    # remove the nlp engine we passed, to use custom label mappings
    registry.remove_recognizer("SpacyRecognizer")

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine,
                              registry=registry,
                              supported_languages=["en"])

    return analyzer

def generate_surrogate(name):
    """Return appropriate surrogate name from text string"""
    if "John" in name:
        return "Jill"
    else:
        return "SURROGATE_NAME"