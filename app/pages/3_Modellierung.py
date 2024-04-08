import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_transformer,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeRegressor, plot_tree


# Take log from target
def norm_target(x):
    return np.log(x + 1)


def renorm_target(y):
    x = np.exp(y) - 1
    return x


def plot_dec_tree(dec_tree, feature_names):
    fig, ax = plt.subplots()
    plot_tree(dec_tree, filled=True, ax=ax)
    return fig


st.set_page_config(page_title="Modellierung")


if "dataset" not in st.session_state:
    st.error("Bitte zuerst einen Datensatz auswählen!")
    st.markdown("[Datenauswahl](Datenauswahl)")
else:
    widget_key = 0
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

    st.markdown("# Modellierung")
    st.sidebar.header("Modellierung")
    with st.expander("Lineare Regression"):
        numeric_transformer = Pipeline(
            steps=[
                ("scale", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_features),
                ("cat", categorical_transformer, cat_features),
            ]
        )
        # Model Pipeline Setup
        linear_model = make_pipeline(
            preprocessor,
            TransformedTargetRegressor(
                regressor=LinearRegression(),
                func=norm_target,
                inverse_func=renorm_target,
            ),
        )

        # Train, Test and Evaluate
        linear_model.fit(X_train, y_train)

        # Evaluate with MAE -> all deviations have the same weight
        y_pred_test = linear_model.predict(X_test)
        y_pred_train = linear_model.predict(X_train)

        MAE_test = mean_absolute_error(y_test, y_pred_test)
        MAE_train = mean_absolute_error(y_train, y_pred_train)
        st.markdown(f"MAE (train): {MAE_train: .2f}")
        st.markdown(f"MAE (test): {MAE_test: .2f}")
        MSE_test = mean_squared_error(y_test, y_pred_test)
        MSE_train = mean_squared_error(y_train, y_pred_train)
        st.markdown(f"RMSE (train): {np.sqrt(MSE_train): .2f}")
        st.markdown(f"RMSE (test):  {np.sqrt(MSE_test): .2f}")

    with st.expander("Entscheidungsbaum"):
        numeric_transformer = Pipeline(
            steps=[
                ("scale", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_features),
                ("cat", categorical_transformer, cat_features),
            ]
        )
        # Model Pipeline Setup
        tree_depth = max_depth_selection = st.select_slider(
            "Baumtiefe",
            options=np.arange(1, 15),
        )
        widget_key += 1

        tree_model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", DecisionTreeRegressor(max_depth=tree_depth)),
            ]
        )

        # Train, Test and Evaluate
        tree_model.fit(X_train, y_train)

        # Evaluate with MAE -> all deviations have the same weight
        y_pred_test = tree_model.predict(X_test)
        y_pred_train = tree_model.predict(X_train)

        MAE_test = mean_absolute_error(y_test, y_pred_test)
        MAE_train = mean_absolute_error(y_train, y_pred_train)
        st.markdown(f"MAE (train): {MAE_train: .2f}")
        st.markdown(f"MAE (test): {MAE_test: .2f}")
        MSE_test = mean_squared_error(y_test, y_pred_test)
        MSE_train = mean_squared_error(y_train, y_pred_train)
        st.markdown(f"RMSE (train): {np.sqrt(MSE_train): .2f}")
        st.markdown(f"RMSE (test):  {np.sqrt(MSE_test): .2f}")
        st.pyplot(
            plot_dec_tree(
                tree_model.named_steps["regressor"],
                feature_names=X.columns,
            ),
        )
