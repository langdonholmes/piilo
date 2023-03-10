import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from nameparser import HumanName
from names_dataset import NameDataset
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.operators import OperatorType

data = Path(__file__).parent.parent.parent / 'data' 
name_table = data / 'ascii_names.parquet'

logger = logging.getLogger('anonymizer')

class NameDatabase(NameDataset):
    '''A wrapper around the names_dataset.NameDataset class.
    '''
    
    def __init__(self) -> None:
        super().__init__()

    def _max(self, name_stats: dict) -> str:
        try:
            g_ = {a: b for a, b in name_stats.items() if b is not None}
            return max(g_, key=g_.get)
        except ValueError:
            return None
       
    def get_gender(self, name: HumanName) -> str:
        '''Return the most frequent gender code for the provided name's 
        first name, or None if a match cannot be found.
        '''
        result = self.first_names.get(name.first)
        return self._max(result.get('gender', None)) if result else None

    def get_country(self, name: HumanName) -> str:
        '''Return the most frequent country code for the provided name's 
        last name, or None if a match cannot be found.
        '''
        result = self.last_names.get(name.last)
        return self._max(result.get('country', None)) if result else None
    
class SurrogateAnonymizer(AnonymizerEngine):
    '''A wrapper around the presidio_anonymizer.AnonymizerEngine class.
    '''
    
    def __init__(self):
        super().__init__()
        self.names_db = NameDatabase()
        self.names_df = pd.read_parquet(name_table)
        
        # keep track of names we have seen
        self.seen_first_names = dict()
        self.seen_last_names = dict()
        
    def get_random_name(
            self,
            country: Optional[str] = None,
            gender: Optional[str] = None
    ) -> pd.Series:
        '''Returns a random name from the database as pd.Series.
        Matches gender and country, if provided.
        :country: ISO country code e.g. 'CO' for Columbia
        :gender: 'M' or 'F'
        '''
        
        names_view = self.names_df
        
        if country:
            names_view = names_view[names_view['country'] == country]
            
        if gender:
            names_view = names_view[names_view['gender'] == gender]
            
        if names_view.size < 25:
            # If we don't have enough names, just return a random sample
            return self.names_df.sample(n=1, weights=self.names_df['count'])
        
        return names_view.sample(n=1, weights=names_view['count'])

    def generate_surrogate(self, original_name: str) -> str:
        '''Generate a surrogate name.
        '''
        
        if original_name == 'PII':
            # Every time we call this function, Presidio will validate it
            # by testing that the function returns a str when the input is
            # 'PII'. We don't need to run below code in this case.
            return 'PII'
        
        # Use nameparser to split the name
        name = HumanName(original_name.strip().title())
        new_name = HumanName()
        gender, country = None, None
        
        # First check if we have seen this name before
        if name.last:
            if name.last in self.seen_last_names:
                logger.info(f'Last name has already been seen.')
                new_name.last = self.seen_last_names[name.last]
            else:
                # Sample last name, attempting to match country.
                country = self.names_db.get_country(name)
                logger.info(f'Country set to {country}')
                new_name.last = self.get_random_name(
                    country=country,
                    )['last'].iloc[0]
                logger.info(f'Last name surrogate is {new_name.last}')
                
        if name.first:
            if name.first in self.seen_first_names:
                logger.info(f'First name has already been seen.')
                new_name.first = self.seen_first_names[name.first]
            else:
                # Sample first name, attempting to match gender and country.
                gender = self.names_db.get_gender(name)
                logger.info(f'Gender set to {gender}')
                new_name.first = self.get_random_name(
                    gender=gender,
                    country=country,
                    )['first'].iloc[0]
                logger.info(f'First name surrogate is {new_name.first}')
            
        logger.info(f'Returning surrogate name {new_name}')
        
        self.seen_first_names[name.first] = new_name.first
        self.seen_last_names[name.last] = new_name.last
                
        return str(new_name)

    def anonymize(
        self,
        text: str,
        analyzer_results: List[RecognizerResult]
        ):
        '''Anonymize identified input using Presidio Anonymizer.
        '''
        
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
    
    anonymizer = SurrogateAnonymizer()
    
    test_names = ['Nora Wang',
                  'John Williams',
                  'John H. Williams',
                  'MJ',
                  '',
                  '(',
                  'Mario Escobar Sanchez',
                  'Jane Fonda Michelle Rousseau',
                  'Sir Phillipe Ricardo de la Sota Mayor',
                  'Anthony REDDIX',
                  'REDDIX, Anthony',
                  ]
    
    for test_name in test_names:
        name = HumanName(test_name.strip().title())
        logger.info(f'{test_name} -> {name.first} + {name.last}')
    
        anonymizer.generate_surrogate(test_name)