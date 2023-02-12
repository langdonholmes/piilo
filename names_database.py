import logging


from names_dataset import NameDataset, NameWrapper


class NameDatabase(NameDataset):
    def __init__(self) -> None:
        super().__init__()
        
        self.logger = logging.getLogger('anonymizer')
    
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