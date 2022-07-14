import matplotlib as mpl
import pandas as pd
import streamlit as st

mpl.style.use("default")

st.set_page_config(page_title="Machine Learning Workflow", layout="wide")
st.markdown("# Übersicht")
st.sidebar.header("Übersicht")
st.sidebar.success("Wähle einen Schritt aus dem Workflow aus.")

if "housing_data" not in st.session_state:
    st.session_state["housing_data"] = pd.read_csv("data/housing.csv")

housing_data = st.session_state["housing_data"]

st.write(f"Number samples: {housing_data.shape[0]}")
st.write(f"Number features: {housing_data.shape[1]}")

st.dataframe(housing_data.head())

# More dimensions
st.map(housing_data)
