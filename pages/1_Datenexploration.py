import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

st.set_page_config(page_title="Datenverständnis")

st.markdown("# Datenverständnis")
st.sidebar.header("Datenverständnis")

if "data" not in st.session_state:
    st.error("Bitte zuerst einen Datensatz auswählen!")
    st.markdown("[Datenauswahl](Datenauswahl)")
else:
    data = st.session_state["data"]
    df = data["frame"]
    
    feature = st.selectbox("Feature", options=df.columns)

    # Univariate Plots
    st.sidebar.markdown("## Univariate Plots")
    st.sidebar.markdown("#### [Histogramme](#histogramme) ")
    st.write("### Histogramme")

    bins = st.select_slider("Anzahl Bins", options=[5, 10, 20, 50, 100])


    @st.cache(allow_output_mutation=True)
    def hist_plot(col: str, bins: int):

        fig, ax = plt.subplots()
        disp = ax.hist(df[col], bins=bins, alpha=0.7, edgecolor="k")
        min_ylim, max_ylim = ax.get_ylim()
        min_xlim, max_xlim = ax.get_xlim()
        x_range = abs(max_xlim - min_xlim)
        ax.axvline(
            df[col].mean(), color="k", linestyle="dashed", linewidth=1
        )
        ax.text(
            df[col].mean() + x_range * 0.02,
            max_ylim * 0.9,
            "Mean: {:.2f}".format(df[col].mean()),
        )
        ax.axvline(
            df[col].median(), color="k", linestyle="solid", linewidth=1
        )
        ax.text(
            df[col].median() + x_range * 0.02,
            max_ylim * 0.8,
            "Median: {:.2f}".format(df[col].median()),
        )
        ax.set_xlabel(col)

        return fig


    st.pyplot(hist_plot(feature, bins))

    st.sidebar.markdown("#### [Box Plots](#box-plots) ")
    st.write("### Box Plots")

    @st.cache(allow_output_mutation=True)
    def box_plot(col: str):

        fig, ax = plt.subplots()
        fig_data = ax.boxplot(
            df[col], bootstrap=1000, autorange=True, showmeans=True
        )

        return fig


    st.pyplot(box_plot(feature))

    # Multivariate Plots
    st.sidebar.markdown("## Multivariate Plots")
    st.sidebar.markdown("#### [Scatter Plots](#scatter-plots) ")
    st.write("### Scatter Plots")

    widget_key = 0
    scatter_x = st.selectbox(
        "X-Achse", options=df.columns, key=widget_key, index=0
    )
    widget_key += 1
    scatter_y = st.selectbox(
        "Y-Achse", options=df.columns, key=widget_key, index=1
    )
    opt_selections = [x for x in df.columns] + [None]
    scatter_color_selection = st.selectbox(
        "Farbe", options=opt_selections, key=widget_key, index=opt_selections.index(None)
    )
    if scatter_color_selection == None:
        scatter_color = None
    elif df.dtypes[scatter_color_selection] == "object":
        scatter_color = labelencoder.fit_transform(df[scatter_color_selection])
    else:
        scatter_color = df[scatter_color_selection]
        
    widget_key += 1
    scatter_size_selection = st.selectbox(
        "Größe", options=opt_selections, key=widget_key, index=opt_selections.index(None)
    )
    widget_key += 1
    scatter_size_ratio = st.select_slider(
        "Größenratio",
        options=[0.5, 1, 2, 5, 10, 20, 50, 100, 200],
        key=widget_key,
    )
    if scatter_size_selection == None:
        scatter_size = 100 / scatter_size_ratio
    elif scatter_size_selection == "ocean_proximity":

        scatter_size = labelencoder.fit_transform(df[scatter_size_selection])

    else:
        scatter_size = df[scatter_size_selection] / scatter_size_ratio


    @st.cache(allow_output_mutation=True)
    def scatter_plot(scatter_x, scatter_y, scatter_color, scatter_size):

        fig, ax = plt.subplots()
        disp = ax.scatter(
            df[scatter_x],
            df[scatter_y],
            alpha=0.1,
            s=scatter_size,
            label=scatter_size,
            c=scatter_color,
            cmap=plt.get_cmap("nipy_spectral"),
        )
        ax.set_xlabel(scatter_x)
        ax.set_ylabel(scatter_y)
        fig.colorbar(disp, ax=ax)
        return fig


    st.pyplot(scatter_plot(scatter_x, scatter_y, scatter_color, scatter_size))
