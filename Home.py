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

st.dataframe(st.session_state["housing_data"].head())


# Scatter plot of geo data
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# Check different distributions
# housing_data.hist(bins=50, figsize=(20,15))
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)

# st.pyplot(fig)

# More dimensions
st.map(housing_data)
