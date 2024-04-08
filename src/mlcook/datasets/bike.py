"""
Definition of the Bike dataset
"""

from pathlib import Path

import pandas as pd

from .base import Dataset


class BikeDataset(Dataset):
    name = "Bike"

    def _load_data(self):
        # dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"
        dataset_folder = Path(__file__).parent.parent.parent.parent / "data"
        local_data = dataset_folder / "SeoulBikeData.csv"
        df: pd.DataFrame = pd.read_csv(local_data, encoding="latin")
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

        y = "Rented Bike Count"

        self.data = df
        self.X = df.drop(columns=[y])
        self.y = df[y]
        self.target = y
        self.descr = """
        # Hintergrundinformationen

        Bike-Sharing-Systeme sind ein Mittel zum Mieten von Fahrrädern, bei denen der Prozess der Mitgliedschaft, des Verleihs und der Fahrradrückgabe über ein Netzwerk von Kioskstandorten in der ganzen Stadt automatisiert wird. Mit diesen Systemen können Menschen ein Fahrrad an einem Ort mieten und es bei Bedarf an einem anderen Ort zurückgeben. Derzeit gibt es weltweit über 500 Bike-Sharing-Programme.

        Die von diesen Systemen generierten Daten machen sie für Forscher attraktiv, da Reisedauer, Abfahrtsort, Ankunftsort und verstrichene Zeit explizit erfasst werden. Bike-Sharing-Systeme fungieren somit als Sensornetzwerk, mit dem die Mobilität in einer Stadt untersucht werden kann. Bei diesem Wettbewerb werden die Teilnehmer gebeten, historische Nutzungsmuster mit Wetterdaten zu kombinieren, um die Fahrradmietnachfrage im Rahmen des Capital Bikeshare-Programms in Washington, DC, zu prognostizieren.

        http://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand


        ## Variable Explanations

        | Variable |Explanation |
        | --- | --- |
        | Date | year-month-day |
        |Rented Bike count | Count of bikes rented at each hour |
        | Hour | Hour of the day |
        | Temperature | Temperature in Celsius |
        | Humidity | % |
        | Windspeed | m/s |
        | Visibility | 10m |
        | Dew point temperature | Celsius |
        | Solar radiation | MJ/m2 |
        | Rainfall | mm |
        | Snowfall | cm |
        | Seasons | Winter, Spring, Summer, Autumn |
        | Holiday | Holiday/No holiday |
        | Functional Day | NoFunc(Non Functional Hours), Fun(Functional hours) |
        """

    @property
    def categorical_features(self):
        return ("Seasons", "Holiday", "Hour", "Functioning Day", "Date")

    @property
    def numerical_features(self):
        return tuple(set(self.data.columns) - set(self.categorical_features))

    def is_regression(self):
        return True
