import streamlit as st
from PIL import Image


st.set_page_config(page_title="Machine Learning Workflow - Home", layout="wide")

st.sidebar.header("Übersicht")
st.sidebar.success("Wähle einen Schritt aus dem Workflow aus.")

st.markdown("# Übersicht")

st.header("CRISP-DM")

img = Image.open("rsc/img/crisp-dm.png")
st.image(img)

st.header("Machine Learning Workflow")

img = Image.open("rsc/img/ml-workflow.jpg")
st.image(img)
