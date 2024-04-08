import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

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

dataset = st.session_state["dataset"]
df = dataset.data

X = dataset.X
y = dataset.y
X = X.select_dtypes(exclude="datetime")

# Data Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=None,
    shuffle=True,
    random_state=42,
)

cat_features = list(dataset.categorical_features)
cat_features.remove("Date")
num_features = list(dataset.numerical_features)

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
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)


train_error = []
test_error = []
max_depth_selection = st.select_slider(
    "Baumtiefe",
    options=np.arange(1, 21),
)


@st.cache_resource
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
