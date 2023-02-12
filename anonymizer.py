from typing import List

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from names_database import NameDatabase

names_db = NameDatabase()

def split_name(original_name: str):
    '''Splits name into parts.
    If one token, assume it is a first name.
    If two tokens, first and last name.
    If three tokens, one first name and two last names.
    If four tokens, two first names and two last names.'''
    names = original_name.split()
    if len(names) == 1:
        return names[0], None
    elif len(names) == 2:
        return names[0], names[1]
    elif len(names) == 3:
        return names[0], ' '.join(names[1:])
    elif len(names) == 4:
        return ' '.join(names[:2]), ' '.join(names[2:])
    else:
        return None, None

def generate_surrogate(original_name: str):
    '''Generate a surrogate name.
    '''
    first_names, last_names = split_name(original_name)
    gender = names_db.get_gender(first_names) if first_names else None
    country = names_db.get_country(last_names) if last_names else None
    
    surrogate_name = ''
    
    name_candidates = names_db.get_random_name(
        gender=gender,
        country=country)
    
    surrogate_name += name_candidates.iloc[0]['first']
    
    if last_names:
        surrogate_name += ' ' + name_candidates.iloc[1]['last']
        
    return surrogate_name

def anonymize(
    anonymizer: AnonymizerEngine,
    text: str,
    analyze_results: List[RecognizerResult]
    ):
    '''Anonymize identified input using Presidio Anonymizer.'''
    
    if not text:
        return
    
    res = anonymizer.anonymize(
        text,
        analyze_results,
        operators={
            'STUDENT': OperatorConfig('custom',
                                      {'lambda': generate_surrogate}),
            'EMAIL_ADDRESS': OperatorConfig('replace',
                                            {'new_value': 'janedoe@aol.com'}),
            'PHONE_NUMBER': OperatorConfig('replace',
                                           {'new_value': '888-888-8888'}),
            'URL': OperatorConfig('replace',
                                  {'new_value': 'aol.com'}),
            }
    )
    
    return res.text

if __name__ == '__main__':
    print(generate_surrogate('Nora Wang'))