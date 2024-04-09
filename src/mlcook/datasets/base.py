import abc

import pandas as pd


class Dataset(metaclass=abc.ABCMeta):
    def __init__(self):
        self.data = pd.DataFrame()
        self.target = ""
        self.X = pd.DataFrame()
        self.y = pd.Series(dtype=str)
        self.descr = ""

        self._load_data()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            raise TypeError("Subclass must define class attribute 'name'")

    @abc.abstractmethod
    def _load_data(self) -> None:
        pass

    @abc.abstractmethod
    def is_regression(self) -> bool:
        pass

    def is_classification(self) -> bool:
        return not self.is_regression()

    @property
    @abc.abstractmethod
    def categorical_features(self) -> tuple[str]:
        pass

    @property
    @abc.abstractmethod
    def numerical_features(self) -> tuple[str]:
        pass

    @property
    def geo_features(self) -> tuple[str]:
        """Return the names of the geolocation features in order latitude, longitude"""
        return tuple()

    @property
    def datetime_features(self):
        return tuple()

    @property
    def features(self) -> tuple[str]:
        return tuple(self.X.columns)

    @property
    def n_features(self) -> int:
        return len(self.X.columns)

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]
