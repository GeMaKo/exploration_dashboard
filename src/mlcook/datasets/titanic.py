import pandas as pd
import seaborn as sns

from .base import Dataset


class TitanicDataset(Dataset):
    name = "Titanic"

    def _load_data(self):
        df: pd.DataFrame = sns.load_dataset("titanic")
        y = "survived"
        self.data = df.drop(columns=["deck"])
        self.X = df.drop(columns=[y, "deck"])
        self.y = df["survived"]
        self.target = "survived"
        self.descr = ""

    @property
    def categorical_features(self):
        return (
            "pclass",
            "sex",
            "embarked",
            "class",
            "who",
            "adult_male",
            "embark_town",
            "alive",
            "alone",
        )

    @property
    def numerical_features(self):
        return tuple(set(self.data.columns) - set(self.categorical_features))

    def is_regression(self):
        return False
