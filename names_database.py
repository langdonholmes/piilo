from names_dataset import NameDataset, NameWrapper
from typing import Optional

class NameDatabase(NameDataset):
    def __init__(self) -> None:
        super().__init__()
        self.names = pd.read_parquet('ascii_fb_names_small.parquet')

    def get_random_name(
            self,
            country: Optional[str] = None,
            gender: Optional[str] = None
    ):
        '''country: ISO country code in 'alpha 2' format
        gender: "M" or "F"
        '''
        names_view = self.names
        if country:
            names_view = names_view[names_view['country'] == country]
        if gender:
            names_view = names_view[names_view['gender'] == gender]
        return names_view.sample(weights=names_view.count)
    
    def get_gender(first_names: str):
        return NameWrapper(self.search(first_names)).gender
    
    def get_country(last_names: str):
        return NameWrapper(self.search(last_names)).country