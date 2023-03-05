'''Main  for PIILO'''

import logging

# assumes piilo is in site-packages
from piilo.engines.analyzer import CustomAnalyzer
from piilo.engines.anonymizer import SurrogateAnonymizer
from piilo.models.anonymize import AnonymizeRequest, AnonymizeResponse

configuration = {
    'nlp_engine_name': 'spacy',
    'models': [
        {'lang_code': 'en', 'model_name': 'en_student_name_detector'}],
}

logger = logging.getLogger('piilo')

logger.info("Loading Custom Presidio Analyzer and Anonymizer...")
analyzer = CustomAnalyzer(configuration)
anonymizer = SurrogateAnonymizer()
logger.info("Loading Successful!")

def analyze(raw_text: str, entities=None, language='en'):
    analyzer_result = analyzer.analyze(raw_text,
                                       entities=entities,
                                       language=language,
                                       )
    return analyzer_result

def anonymize(raw_text: str, entities=None, language='en') -> str:
    
    analyzer_result = analyze(raw_text, entities, language)
    
    return anonymizer.anonymize(raw_text,
                                analyzer_result)

def get_anonymize(anon_req: AnonymizeRequest) -> AnonymizeResponse:
    
    anonymizer_result = anonymize(anon_req.raw_text,
                                  entities=anon_req.entities,
                                  language=anon_req.language,
                                  )
    
    anonymize_response = AnonymizeResponse(
        anonymized_text=anonymizer_result
        )
    
    return anonymize_response

if __name__ == "__main__":
    pass