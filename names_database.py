from pathlib import Path
from typing import Optional

import pandas as pd
from names_dataset import NameDataset, NameWrapper

name_table = Path('data', 'ascii_names.parquet')

class NameDatabase(NameDataset):
    def __init__(self) -> None:
        super().__init__()
        self.names = pd.read_parquet(name_table)

    def get_random_name(
            self,
            country: Optional[str] = None,
            gender: Optional[str] = None
    ):
        '''country: ISO country code in 'alpha 2' format
        gender: 'M' or 'F'
        returns two rows of the names dataframe
        '''
        names_view = self.names
        if country:
            names_view = names_view[names_view['country'] == country]
        if gender:
            names_view = names_view[names_view['gender'] == gender]
        if names_view.size < 25:
            return self.names.sample(n=2, weights=self.names['count'])
        return names_view.sample(n=2, weights=names_view['count'])
    
    def search(self, name: str):
        key = name.strip().title()
        fn = self.first_names.get(key) if self.first_names is not None else None
        ln = self.last_names.get(key) if self.last_names is not None else None
        return {'first_name': fn, 'last_name': ln}
       
    def get_gender(self, first_names: str):
        return NameWrapper(self.search(first_names)).gender

    def get_country(self, last_names: str):
        return NameWrapper(self.search(last_names)).country