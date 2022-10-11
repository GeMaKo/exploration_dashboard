import pandas as pd
from sklearn.datasets import fetch_california_housing

from .base import Dataset


class CaliforniaDataset(Dataset):
    name = "California Housing"

    def _load_data(self):
        bunch = fetch_california_housing(as_frame=True)
        target_name = bunch["target_names"][0]

        df: pd.DataFrame = bunch["frame"]
        self.data = df
        self.X = bunch["data"]
        self.target = target_name
        self.y = df[target_name]
        self.descr = bunch["DESCR"]

    @property
    def categorical_features(self):
        return tuple()

    @property
    def numerical_features(self):
        return tuple(self.X.columns)

    @property
    def geo_features(self):
        return ("Latitude", "Longitude")

    def is_regression(self):
        return True
