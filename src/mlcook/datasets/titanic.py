import pandas as pd
import seaborn as sns

from .base import Dataset


class TitanicDataset(Dataset):
    name = "Titanic"

    def _load_data(self):
        df: pd.DataFrame = sns.load_dataset("titanic")(as_frame=True)
        df = df.rename(columns={"target": "class"})
        y = "survived"
        self.data = df
        self.X = df.drop(columns=[y])
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
