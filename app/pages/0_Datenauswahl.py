import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from mlcook.datasets.manager import DatasetManager
from mlcook.datasets.base import Dataset

st.set_page_config(page_title="Machine Learning Workflow - Datenauswahl", layout="wide")

st.markdown("### Auswahl Datensatz")

# SIDEBAR MENU
st.sidebar.header("Datenauswahl")
st.sidebar.markdown("#### [Auswahl Datensatz](#auswahl-datensatz)")

# MAIN PAGE
manager = DatasetManager()

def load_data_into_session(dataset: str):
    st.session_state["dataset"] = manager.init_dataset(dataset)


if "dataset" in st.session_state:
    dataset = st.session_state["dataset"]
    default_index = list(manager.datasets.keys()).index(dataset.name)
else:
    default_index = 0

dataset = st.selectbox(
    "Datensatz", options=manager.datasets.keys(), index=default_index
)
load_data_into_session(dataset)

st.markdown("### Weitere Infos")

dataset: Dataset = st.session_state["dataset"]
df: pd.DataFrame = dataset.data

st.write(f"Anzahl Beispiele: {df.shape[0]}")
st.write(f"Anzahl Dimensionen: {df.shape[1]}")

st.dataframe(df.head())

with st.expander("Beschreibung anzeigen"):
    st.write(dataset.descr)

if dataset == "California Housing":
    df.columns = [col.lower() for col in df.columns]
    # More dimensions
    st.map(df)

if dataset == "Digits":
    fig, ax = plt.subplots(1, 5)

    for i, img in enumerate(df.drop(columns=["target"]).iloc[:5, :].values):
        ax[i].imshow(img.reshape(8, 8), cmap=plt.get_cmap("binary"))
        plt.setp(ax[i], xticks=range(8), yticks=range(8))

    plt.tight_layout()

    st.pyplot(fig)

if dataset == "Olivetti Faces":
    fig, ax = plt.subplots(1, 2)

    for i, img in enumerate(df.drop(columns=["target"]).iloc[:2, :].values):
        ax[i].imshow(img.reshape(64, 64))
        plt.setp(ax[i], xticks=range(64)[::8], yticks=range(64)[::8])

    plt.tight_layout()

    st.pyplot(fig)
