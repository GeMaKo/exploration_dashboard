import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

st.set_page_config(page_title="Datenvorbereitung")

st.markdown("# Datenvorbereitung")
st.sidebar.header("Datenvorbereitung")


def box_plot(df, col, show_outlier: bool = True):
    fig, ax = plt.subplots()
    fig_data = ax.boxplot(
        df[col],
        bootstrap=1000,
        autorange=True,
        showmeans=True,
        showfliers=show_outlier,
    )
    ax.set_xticklabels(col, rotation=30)
    return fig


if "dataset" not in st.session_state:
    st.error("Bitte zuerst einen Datensatz auswählen!")
    st.markdown("[Datenauswahl](Datenauswahl)")
else:
    dataset = st.session_state["dataset"]
    df = dataset.data.loc[:, dataset.numerical_features]

    st.write("### Normalization")
    st.dataframe(df.head())
    st.write("#### Raw data")
    dist_plot = st.pyplot(box_plot(df, df.columns.tolist(), True))

    st.write("#### Scaled data")
    df_standard_scaler = df.copy()
    df_standard_scaler[df_standard_scaler.columns] = StandardScaler().fit_transform(df)
    st.dataframe(df_standard_scaler.head())
    dist_plot = st.pyplot(box_plot(df_standard_scaler, df.columns, False))
    df_minmax_scaler = df.copy()
    df_minmax_scaler[df_minmax_scaler.columns] = MinMaxScaler().fit_transform(df)
    st.dataframe(df_minmax_scaler.head())
    dist_plot = st.pyplot(box_plot(df_minmax_scaler, df.columns, True))
