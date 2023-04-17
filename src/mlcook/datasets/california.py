import pandas as pd
from sklearn.datasets import fetch_california_housing

from .base import Dataset


class CaliforniaDataset(Dataset):
    name = "California Housing"

    def _load_data(self):
        bunch = fetch_california_housing(as_frame=True)
        local_data = pd.read_csv("data/housing.csv")
        local_data = local_data.dropna()
        target_name = "median_house_value"

        df: pd.DataFrame = local_data
        df = df.sample(10000)
        self.data = df
        self.X = df.loc[:, df.columns != target_name]
        self.target = target_name
        self.y = df[target_name]
        self.descr = bunch["DESCR"]

    @property
    def categorical_features(self):
        return ("ocean_proximity",)

    @property
    def numerical_features(self):
        num_cols = self.X.loc[:, self.X.columns != "ocean_proximity"].columns
        return tuple(num_cols)

    @property
    def geo_features(self):
        return ("latitude", "longitude")

    def is_regression(self):
        return True
