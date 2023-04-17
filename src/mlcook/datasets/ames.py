"""
Definition of the Ames Housing dataset
"""
from pathlib import Path
import pandas as pd
from .base import Dataset


class AmesDataset(Dataset):
    name = "Ames"

    def _load_data(self):
        root_url = dataset_url = Path(__file__).parent.parent.parent.parent / "data"
        dataset_url = root_url / "ames.csv"
        description_url = root_url / "ames_description.txt"
        df: pd.DataFrame = pd.read_csv(
            dataset_url, encoding="latin1", sep="\t", na_values="NA"
        )
        y = "SalePrice"

        df = df.drop(columns=["PID", "Order"])

        self.data = df
        self.X = df.drop(columns=[y])
        self.y = df[y]
        self.target = y
        with open(description_url, "r", encoding="latin1") as fhandle:
            self.descr = fhandle.read()

    @property
    def categorical_features(self):
        return (
            "MS SubClass",
            "MS Zoning",
            "Street",
            "Alley",
            "Lot Shape",
            "Land Contour",
            "Utilities",
            "Lot Config",
            "Land Slope",
            "Neighborhood",
            "Condition 1",
            "Condition 2",
            "Bldg Type",
            "House Style",
            "Overall Qual",
            "Overall Cond",
            "Roof Style",
            "Roof Matl",
            "Exterior 1st",
            "Exterior 2nd",
            "Mas Vnr Type",
            "Mas Vnr Area",
            "Exter Qual",
            "Exter Cond",
            "Foundation",
            "Bsmt Qual",
            "Bsmt Cond",
            "Bsmt Exposure",
            "BsmtFin Type 1",
            "BsmtFin Type 2",
            "Heating",
            "Heating QC",
            "Central Air",
            "Electrical",
            "KitchenQual",
            "Functional",
            "FireplaceQu",
            "Garage Type",
            "Garage Finish",
            "Garage Qual",
            "Garage Cond",
            "Paved Drive",
            "Pool QC",
            "Fence",
            "Misc Feature",
            "Sale Type",
            "Sale Condition",
        )

    @property
    def numerical_features(self):
        return tuple(set(self.data.columns) - set(self.categorical_features))

    def is_regression(self):
        return True
