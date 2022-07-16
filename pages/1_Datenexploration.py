import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder

sns.set_style("darkgrid")
sns.set_palette("tab10")

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

    feature = st.selectbox("Variable", options=df.columns)
    hue_options = list(df.columns) + [None]
    hue_var = st.selectbox(
        "Trennvariable", options=hue_options, index=hue_options.index(None)
    )

    # Univariate Plots
    st.sidebar.markdown("## Univariate Plots")
    st.sidebar.markdown("#### [Histogramme](#histogramme) ")
    st.write("### Histogramme")

    bins = st.select_slider("Anzahl Bins", options=[5, 10, 20, 50, 100])
    kde = st.checkbox("Dichteschätzer")
    log_scale = st.checkbox("Log. Skalierung")

    def hist_plot(
        col: str, bins: int, kde: bool = False, log_scale: bool = False, hue: str = None
    ):

        fig, ax = plt.subplots()

        if hue is None:
            sns.histplot(
                df,
                x=col,
                bins=bins,
                ax=ax,
                kde=kde,
                stat="count",
                log_scale=log_scale,
            )
        else:
            sns.histplot(
                df,
                x=col,
                bins=bins,
                ax=ax,
                kde=kde,
                stat="count",
                log_scale=log_scale,
                hue=df[hue].astype(str),
                multiple="layer",
            )
        min_ylim, max_ylim = ax.get_ylim()
        min_xlim, max_xlim = ax.get_xlim()
        x_range = abs(max_xlim - min_xlim)
        ax.axvline(df[col].mean(), color="k", linestyle="dashed", linewidth=1)
        ax.text(
            df[col].mean() + x_range * 0.02,
            max_ylim * 0.9,
            "Mean: {:.2f}".format(df[col].mean()),
        )
        ax.axvline(df[col].median(), color="k", linestyle="solid", linewidth=1)
        ax.text(
            df[col].median() + x_range * 0.02,
            max_ylim * 0.8,
            "Median: {:.2f}".format(df[col].median()),
        )
        ax.set_xlabel(col)

        return fig

    st.pyplot(hist_plot(feature, bins, kde, log_scale, hue_var))

    st.sidebar.markdown("#### [Box Plots](#box-plots) ")
    st.write("### Box Plots")

    show_outlier = st.checkbox("Zeige Outlier", value=True)

    @st.cache(allow_output_mutation=True)
    def box_plot(col: str, show_outlier: bool = True):

        fig, ax = plt.subplots()
        fig_data = ax.boxplot(
            df[col],
            bootstrap=1000,
            autorange=True,
            showmeans=True,
            showfliers=show_outlier,
        )

        return fig

    st.pyplot(box_plot(feature, show_outlier))

    # Multivariate Plots
    st.sidebar.markdown("## Multivariate Plots")
    st.sidebar.markdown("#### [Scatter Plots](#scatter-plots) ")
    st.write("### Scatter Plots")

    widget_key = 0
    scatter_x = st.selectbox("X-Achse", options=df.columns, key=widget_key, index=0)
    widget_key += 1
    scatter_y = st.selectbox("Y-Achse", options=df.columns, key=widget_key, index=1)
    opt_selections = [x for x in df.columns] + [None]
    scatter_color = st.selectbox(
        "Farbe",
        options=opt_selections,
        key=widget_key,
        index=opt_selections.index(None),
    )
    widget_key += 1
    scatter_size = st.selectbox(
        "Größe",
        options=opt_selections,
        key=widget_key,
        index=opt_selections.index(None),
    )
    widget_key += 1
    alpha = st.slider("Alpha", min_value=0.01, max_value=1.0, value=0.7)

    def scatter_plot(
        scatter_x: str,
        scatter_y: str,
        scatter_color: str,
        scatter_size: str,
        alpha: float,
    ):

        fig, ax = plt.subplots()

        if scatter_color is None:
            sns.scatterplot(
                data=df,
                x=scatter_x,
                y=scatter_y,
                alpha=alpha,
                size=scatter_size,
                ax=ax,
            )
        else:
            sns.scatterplot(
                data=df,
                x=scatter_x,
                y=scatter_y,
                alpha=alpha,
                hue=df[scatter_color].astype(str),
                size=scatter_size,
                ax=ax,
            )

        return fig

    st.pyplot(scatter_plot(scatter_x, scatter_y, scatter_color, scatter_size, alpha))
