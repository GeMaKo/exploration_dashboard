import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Datenvorbereitung")

st.markdown("# Datenvorbereitung")
st.sidebar.header("Datenvorbereitung")


def box_plot(df, col: str, show_outlier: bool = True):
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
    df = dataset.data

    st.write("### Normalization")
    st.dataframe(df.head())
    st.write("#### Raw data")
    dist_plot = st.pyplot(df, box_plot(df.columns, False))
    scale = make_pipeline(StandardScaler())

    st.write("#### Scaled data")
    df_scaled = scale.fit(df[dataset.features], df[dataset.target])
    dist_plot = st.pyplot(df_scaled, box_plot(df.columns, False))
