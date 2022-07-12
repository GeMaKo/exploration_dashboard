import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

st.set_page_config(page_title="Modellierung")

st.markdown("# Modellierung")
st.sidebar.header("Modellierung")

housing_data = pd.read_csv("data/housing.csv")
