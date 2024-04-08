import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

st.set_page_config(page_title="Modellierung")

st.markdown("# Modellierung")
st.sidebar.header("Modellierung")

if "housing_data" not in st.session_state:
    st.session_state["housing_data"] = pd.read_csv("data/housing.csv")

housing_data = st.session_state["housing_data"]
