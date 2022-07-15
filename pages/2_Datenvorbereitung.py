import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

labelencoder = LabelEncoder()

st.set_page_config(page_title="Datenvorbereitung")

st.markdown("# Datenvorbereitung")
st.sidebar.header("Datenvorbereitung")

if "housing_data" not in st.session_state:
    st.session_state["housing_data"] = pd.read_csv("data/housing.csv")
    
housing_data = st.session_state["housing_data"]
