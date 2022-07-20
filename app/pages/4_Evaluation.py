import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

labelencoder = LabelEncoder()

st.set_page_config(page_title="Evaluation")

st.markdown("# Evaluation")
st.sidebar.header("Evaluation")

if "housing_data" not in st.session_state:
    st.session_state["housing_data"] = pd.read_csv("data/housing.csv")

housing_data = st.session_state["housing_data"]

housing_labels = housing_data["median_house_value"].copy()
housing_features = housing_data.drop("median_house_value", axis=1).copy()

from sklearn.model_selection import train_test_split


categorical_features = ["ocean_proximity"]
numeric_features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

X_train, X_test, y_train, y_test = train_test_split(
    housing_features, housing_labels, test_size=0.2, random_state=42
)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


train_error = []
test_error = []
max_depth_selection = st.select_slider(
    "Baumtiefe",
    options=np.arange(1, 21),
)


@st.cache(allow_output_mutation=True)
def overfitting_plot(max_depth_selection):
    for max_depth in range(1, max_depth_selection):
        # DT = DecisionTreeRegressor(max_depth=max_depth)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", DecisionTreeRegressor(max_depth=max_depth)),
            ]
        )
        pipeline.fit(X_train, y_train)
        train_predictions = pd.Series(pipeline.predict(X_train), index=X_train.index)
        test_predictions = pd.Series(pipeline.predict(X_test), index=X_test.index)
        train_error.append(np.sqrt(mean_squared_error(y_train, train_predictions)))
        test_error.append(np.sqrt(mean_squared_error(y_test, test_predictions)))

    fig, ax = plt.subplots()
    ax.plot(train_error)
    ax.plot(test_error)
    ax.legend(["Train error", "Test error"])
    ax.set_ylabel("Error")
    ax.set_xlabel("Baumtiefe")
    return fig


st.pyplot(overfitting_plot(max_depth_selection))
