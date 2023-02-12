import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.operators import OperatorType

from names_database import NameDatabase

name_table = Path('data', 'ascii_names.parquet')

logger = logging.getLogger('anonymizer')


class surrogate_anonymizer(AnonymizerEngine):
    def __init__(self):
        super().__init__()
        self.names_db = NameDatabase()
        self.names_df = pd.read_parquet(name_table)
        
        
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

    def split_name(self, original_name: str):
        '''Splits name into parts.
        If one token, assume it is a first name.
        If two tokens, first and last name.
        If three tokens, one first name and two last names.
        If four tokens, two first names and two last names.'''
        names = original_name.split()
        if len(names) == 1:
            logger.info(f'Splitting to 1 first name: {names}')
            return names[0], None
        elif len(names) == 2:
            logger.info(f'Splitting to 1 first name, 1 last name: {names}')
            return names[0], names[1]
        elif len(names) == 3:
            logger.info(f'Splitting to 1 first name, 2 last names: {names}')
            return names[0], ' '.join(names[1:])
        elif len(names) == 4:
            logger.info(f'Splitting to 2 first names and 2 last names: {names}')
            return ' '.join(names[:2]), ' '.join(names[2:])
        else:
            logger.info(f'Splitting failed, do not match gender/country: {names}')
            return None, None

    def generate_surrogate(self, original_name: str):
        '''Generate a surrogate name.
        '''
        first_names, last_names = self.split_name(original_name)
        gender = self.names_db.get_gender(first_names) if first_names else None
        logger.debug(f'Gender set to {gender}')
        country = self.names_db.get_country(last_names) if last_names else None
        logger.debug(f'Country set to {country}')
        
        surrogate_name = ''
        
        name_candidates = self.get_random_name(gender=gender, country=country)
        
        surrogate_name += name_candidates.iloc[0]['first']
        logger.info(f'First name surrogate is {surrogate_name}')
        
        if last_names:
            logger.info(f'Combining with {name_candidates.iloc[1]["last"]}')
            surrogate_name += ' ' + name_candidates.iloc[1]['last']
            
        logger.info(f'Returning surrogate name {surrogate_name}')
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