import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from nameparser import HumanName
from names_dataset import NameDataset, NameWrapper
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.operators import OperatorType

name_table = Path(__file__).parent.parent / 'data' / 'ascii_names.parquet'

logger = logging.getLogger('anonymizer')

class NameDatabase(NameDataset):
    def __init__(self) -> None:
        super().__init__()
    
    def search(self, name: str) -> dict:
        '''Returns all entries associated with a name string.
        The name string can be multiple tokens. 
        Both first and last names will be matched.
        '''
        key = name.strip().title()
        fn = self.first_names.get(key) if self.first_names is not None else None
        ln = self.last_names.get(key) if self.last_names is not None else None
        return {'first_name': fn, 'last_name': ln}
       
    def get_gender(self, first_names: str) -> str:
        '''Return the most frequent gender code for a specific last name,
        or None if a match cannot be found.
        '''
        gender = NameWrapper(self.search(first_names)).gender
        return gender if gender else None

    def get_country(self, last_names: str) -> str:
        '''Return the most frequent country code for a specific last name,
        or None if a match cannot be found.
        '''
        country = NameWrapper(self.search(last_names)).country
        return country if country else None
    
class surrogate_anonymizer(AnonymizerEngine):
    def __init__(self):
        super().__init__()
        self.names_db = NameDatabase()
        self.names_df = pd.read_parquet(name_table)
        
        # keep track of names we have seen
        self.seen_names = dict()
        
    def get_random_name(
            self,
            country: Optional[str] = None,
            gender: Optional[str] = None
    ) -> pd.DataFrame:
        '''Returns two random names from the database as a DataFrame.
        Both rows match gender and country, if provided.
        :country: ISO country code e.g. "CO" for Columbia
        :gender: 'M' or 'F'
        returns two rows of the names dataframe
        '''
        names_view = self.names_df
        if country:
            names_view = names_view[names_view['country'] == country]
        if gender:
            names_view = names_view[names_view['gender'] == gender]
        if names_view.size < 25:
            return self.names_df.sample(n=2, weights=self.names_df['count'])
        return names_view.sample(n=2, weights=names_view['count'])

    def generate_surrogate(self, original_name: str) -> str:
        '''Generate a surrogate name.
        '''
        if original_name == 'PII':
            # Every time we call this function, Presidio will validate it
            # by testing that the function returns a str when the input is
            # 'PII'. Bypass this test.
            return 'PII'
        
        # If we have seen this name before, return the same surrogate
        if original_name in self.seen_names:
            return self.seen_names[original_name]
        
        # Use nameparser to split the name
        name = HumanName(original_name)
        
        gender = self.names_db.get_gender(name.first) if name.first else None
        logger.info(f'Gender set to {gender}')
        country = self.names_db.get_country(name.last) if name.last else None
        logger.info(f'Country set to {country}')
        
        surrogate_name = ''
        
        name_candidates = self.get_random_name(gender=gender, country=country)
        
        surrogate_name += name_candidates.iloc[0]['first']
        logger.info(f'First name surrogate is {surrogate_name}')
        
        if name.last:
            logger.info(f'Last name surrogate is {name_candidates.iloc[1]["last"]}')
            surrogate_name += ' ' + name_candidates.iloc[1]['last']
            
        logger.info(f'Returning surrogate name {surrogate_name}')
        
        self.seen_names[original_name] = surrogate_name
                
        return surrogate_name

    def anonymize(
        self,
        text: str,
        analyzer_results: List[RecognizerResult]
        ):
        '''Anonymize identified input using Presidio Anonymizer.'''
        
        if not text:
            return

        analyzer_results = self._remove_conflicts_and_get_text_manipulation_data(
            analyzer_results
        )
                
        operators = self._AnonymizerEngine__check_or_add_default_operator(
            {
            'STUDENT': OperatorConfig('custom',
                                      {'lambda': self.generate_surrogate}),
            'EMAIL_ADDRESS': OperatorConfig('replace',
                                            {'new_value': 'janedoe@aol.com'}),
            'PHONE_NUMBER': OperatorConfig('replace',
                                           {'new_value': '888-888-8888'}),
            'URL': OperatorConfig('replace',
                                  {'new_value': 'aol.com'}),
            }
        )
        
        res = self._operate(text,
                            analyzer_results,
                            operators,
                            OperatorType.Anonymize)
                
        return res.text

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    anonymizer = surrogate_anonymizer()
    
    test_names = ['Nora Wang',
                  'MJ',
                  '',
                  '(',
                  'Mario Escobar Sanchez',
                  'Jane Fonda Michelle Rousseau',
                  'Sir Phillipe Ricardo de la Sota Mayor']
    
    for name in test_names:
        anonymizer.generate_surrogate(name)