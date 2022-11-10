import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Datenvorbereitung")

st.markdown("# Datenvorbereitung")
st.sidebar.header("Datenvorbereitung")


def box_plot(col: str, show_outlier: bool = True):
    fig, ax = plt.subplots()
    fig_data = ax.boxplot(
        df[col],
        bootstrap=1000,
        autorange=True,
        showmeans=True,
        showfliers=show_outlier,
    )

    return fig


if "dataset" not in st.session_state:
    st.error("Bitte zuerst einen Datensatz auswählen!")
    st.markdown("[Datenauswahl](Datenauswahl)")
else:
    dataset = st.session_state["dataset"]
    df = dataset.data

    st.write("### Normalization")
    st.dataframe(df.head())
    st.pyplot(box_plot(df.columns, False))
