import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

MAPBOX_TOKEN = load_dotenv("MAPBOX_TOKEN")


mpl.style.use("default")


st.set_page_config(page_title="Machine Learning Workflow", layout="wide")
st.markdown("# Übersicht")
st.sidebar.header("Übersicht")
st.sidebar.success("Wähle einen Schritt aus dem Workflow aus.")

housing_data = pd.read_csv("data/housing.csv")
st.dataframe(housing_data.head())


# Scatter plot of geo data
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# Check different distributions
# housing_data.hist(bins=50, figsize=(20,15))
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)

# st.pyplot(fig)

# More dimensions
st.map(housing_data)
