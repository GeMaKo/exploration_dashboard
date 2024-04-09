"""
Definition of the Melbourne Housing Market dataset
"""

from pathlib import Path

import pandas as pd

from .base import Dataset


class MelbourneDataset(Dataset):
    name = "Melbourne"

    def _load_data(self):
        dataset_folder = Path(__file__).parent.parent.parent.parent / "data"
        local_data = dataset_folder / "Melbourne_housing_FULL.csv"
        df: pd.DataFrame = pd.read_csv(local_data, na_values="NA")
        y = "Price"
        df.dropna(subset=[y], inplace=True)
        df = df.sample(10000)

        # df = df.drop(columns=["PID", "Order"])

        self.data = df
        self.X = df.drop(columns=[y])
        self.y = df[y]
        self.target = y

    @property
    def categorical_features(self):
        return (
            "Type",
            "Method",
            "Regionname",
            "CouncilArea",
            "Date",
        )

    @property
    def numerical_features(self):
        return (
            "Rooms",
            "Bedroom2",
            "Bathroom",
            "Propertycount",
            "Distance",
            "YearBuilt",
            "Lattitude",
            "Longtitude",
        )

    @property
    def features(self):
        return self.categorical_features + self.numerical_features

    @property
    def geo_features(self):
        return ("Lattitude", "Longtitude")

    @property
    def datetime_features(self):
        return tuple("Date")

    def is_regression(self):
        return True
