from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer import RecognizerResult

def retrieve_name_records():
    """Read in a table of names with gender and country code fields."""
    pass

def generate_surrogate(name):
    """Return appropriate surrogate name from text string"""
    if "John" in name:
        return "Jill"
    else:
        return "SURROGATE_NAME"

def anonymize(
    anonymizer: AnonymizerEngine,
    text: str,
    analyze_results: list[RecognizerResult]
    ):
    """Anonymize identified input using Presidio Anonymizer."""
    
    if not text:
        return
    
    res = anonymizer.anonymize(
        text,
        analyze_results,
        operators={
            "STUDENT": OperatorConfig("custom", {"lambda": generate_surrogate}),
            "EMAIL_ADDRESS": OperatorConfig("replace",  {"new_value": "janedoe@aol.com"}),
            "PHONE_NUMBER": OperatorConfig("replace",  {"new_value": "888-888-8888"}),
            "URL": OperatorConfig("replace",  {"new_value": "aol.com"}),
            }
    )
    
    return res.text