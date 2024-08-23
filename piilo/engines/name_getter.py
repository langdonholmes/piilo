from nameparser import HumanName
from names_dataset import NameDataset


class NameGetter(NameDataset):
    """A wrapper around the names_dataset.NameDataset class.
    NameDataset uses some data structures that are optimized for finding
    the most likely gender/country of a name.
    """

    def __init__(self) -> None:
        super().__init__()

    def _max(self, name_stats: dict) -> str:
        try:
            g_ = {a: b for a, b in name_stats.items() if b is not None}
            return max(g_, key=g_.get)
        except ValueError:
            return None

    def get_gender(self, name: HumanName) -> str:
        """Return the most frequent gender code for the provided name's
        first name, or None if a match cannot be found.
        """
        result = self.first_names.get(name.first)
        return self._max(result.get("gender", None)) if result else None

    def get_country(self, name: HumanName) -> str:
        """Return the most frequent country code for the provided name's
        last name, or None if a match cannot be found.
        """
        result = self.last_names.get(name.last)
        return self._max(result.get("country", None)) if result else None