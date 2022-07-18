import pandas as pd
from sklearn.datasets import load_iris

from .base import Dataset


class IrisDataset(Dataset):   
    name = "Iris" 
        
    def _load_data(self):
        bunch = load_iris(as_frame=True)
        target_names = bunch["target_names"]
        
        df: pd.DataFrame = bunch["frame"]
        df = df.rename(columns={"target": "class"})
        df["class"] = df["class"].replace({i:n for i,n in zip(df["class"].unique(), target_names)})
        self.data = df
        self.X = bunch["data"]
        self.y = df["class"]
        self.target = "class"
        self.descr = bunch["DESCR"]

    def get_categorical_features(self):
        return tuple()

    def get_numerical_features(self):
        return tuple(self.X.columns)

    def is_regression(self):
        return False
    
    def is_classification(self):
        return True