from typing import Callable
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import sklearn.datasets

mpl.style.use("default")

st.set_page_config(page_title="Machine Learning Workflow - Datenauswahl", layout="wide")

st.markdown("### Auswahl Datensatz")

DATASET_LOADER = {
    "Iris": sklearn.datasets.load_iris,
    "Diabetes": sklearn.datasets.load_diabetes,
    "Digits": sklearn.datasets.load_digits,
    "Wine": sklearn.datasets.load_wine,
    "Breast Cancer Wisconsin": sklearn.datasets.load_breast_cancer,
    "Olivetti Faces": sklearn.datasets.fetch_olivetti_faces,
    "Forest covertypes": sklearn.datasets.fetch_covtype,
    "California Housing": sklearn.datasets.fetch_california_housing,
}

# SIDEBAR MENU
st.sidebar.header("Datenauswahl")
st.sidebar.markdown("#### [Auswahl Datensatz](#auswahl-datensatz)")


def load_data_into_session(dataset: str):
    loader = DATASET_LOADER[dataset]
    if dataset != "Olivetti Faces":
        st.session_state["data"] = loader(as_frame=True)
        st.session_state["dataset"] = dataset
    else:
        data = loader(shuffle=True)
        df = pd.DataFrame(data["data"])
        df["target"] = data["target"]
        st.session_state["data"] = data
        st.session_state["data"]["frame"] = df

if "dataset" in st.session_state:
    dataset = st.session_state["dataset"]
    default_index = list(DATASET_LOADER.keys()).index(dataset)
else: 
    default_index = 0

dataset = st.selectbox("Datensatz", options=DATASET_LOADER.keys(), index=default_index)
load_data_into_session(dataset)

st.markdown("### Weitere Infos")

data = st.session_state["data"]
df: pd.DataFrame = data["frame"]

st.write(f"Number samples: {df.shape[0]}")
st.write(f"Number features: {df.shape[1]}")

st.dataframe(df.head())

with st.expander("Beschreibung anzeigen"):
    st.write(data["DESCR"])

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
