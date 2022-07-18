import abc

import pandas as pd


class Dataset(metaclass=abc.ABCMeta):
    def __init__(self):
        self.data = pd.DataFrame()
        self.target = ""
        self.X = pd.DataFrame()
        self.y = pd.Series()
        self.descr = ""

        self._load_data()
        
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'name'):
            raise TypeError("Subclass must define class attribute 'name'")

    @abc.abstractclassmethod
    def _load_data(self) -> None:
        pass

    @abc.abstractclassmethod
    def get_categorical_features(self) -> tuple[str]:
        pass

    @abc.abstractclassmethod
    def get_numerical_features(self) -> tuple[str]:
        pass

    def has_geolocation(self) -> bool:
        return (
            set(self.data.columns) & set(["lat", "lon", "latitude", "longitude"])
            != set()
        )

    @property
    def features(self) -> tuple[str]:
        return tuple(self.X.columns)

    @property
    def n_features(self) -> int:
        return len(self.X.columns)

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]
