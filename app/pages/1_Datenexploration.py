import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

sns.set_style("darkgrid")
sns.set_palette("tab10")


def hist_plot(
    col: str, bins: int, kde: bool = False, log_scale: bool = False, hue: str = None
):
    fig, ax = plt.subplots()

    hue_values = df[hue].astype(str) if hue is not None else None

    sns.histplot(
        df,
        x=col,
        bins=bins,
        ax=ax,
        kde=kde,
        stat="count",
        log_scale=log_scale,
        hue=hue_values,
        multiple="stack",
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


def box_plot(col: str, group: str, hue: str):
    fig, ax = plt.subplots()
    fig_data = sns.boxplot(
        data=df,
        x=group,
        y=col,
        hue=hue,
        ax=ax,
    )
    return fig


def scatter_plot(
    scatter_x: str,
    scatter_y: str,
    scatter_color: str,
    scatter_size: str,
    alpha: float,
):
    fig, ax = plt.subplots()

    if scatter_color is None:
        hue_values = None
    else:
        hue_values = scatter_color

    sns.scatterplot(
        data=df,
        x=scatter_x,
        y=scatter_y,
        alpha=alpha,
        hue=hue_values,
        size=scatter_size,
        ax=ax,
    )
    return fig


def kde_plot(
    kde_x: str,
    kde_y: str,
    kde_color: str,
    kde_fill: bool,
):
    fig, ax = plt.subplots()

    sns.kdeplot(
        data=df,
        x=kde_x,
        y=kde_y,
        hue=kde_color,
        fill=kde_fill,
        ax=ax,
    )
    return fig


st.set_page_config(page_title="Datenverständnis")

st.markdown("# Datenverständnis")
st.sidebar.header("Datenverständnis")

if "dataset" not in st.session_state:
    st.error("Bitte zuerst einen Datensatz auswählen!")
    st.markdown("[Datenauswahl](Datenauswahl)")
else:
    dataset = st.session_state["dataset"]
    df = dataset.data
    opt_selections = dataset.features
    opt_selections += (dataset.target,)
    opt_selections += (None,)

    feat_selections = dataset.numerical_features
    if dataset.is_regression():
        feat_selections += (dataset.target,)

    hue_options = dataset.categorical_features
    if dataset.is_classification():
        hue_options += (dataset.target,)
    hue_options += (None,)

    widget_key = 0
    # Univariate Plots
    st.sidebar.markdown("## Univariate Plots")
    st.sidebar.markdown("#### [Histogramme](#histogramme) ")
    st.write("### Histogramme")
    with st.expander("Histogramme"):
        feature = st.selectbox(
            "Variable",
            options=feat_selections,
            key=widget_key,
        )
        widget_key += 1
        hue_var = st.selectbox(
            "Trennvariable",
            options=hue_options,
            index=hue_options.index(None),
            key=widget_key,
        )
        widget_key += 1
        bins = st.select_slider(
            "Anzahl Bins",
            options=[5, 10, 20, 50, 100],
            key=widget_key,
        )
        widget_key += 1
        kde = st.checkbox(
            "Dichteschätzer",
            value=False,
            key=widget_key,
        )
        widget_key += 1
        log_scale = st.checkbox(
            "Log. Skalierung",
            value=False,
            key=widget_key,
        )
        widget_key += 1
        st.pyplot(hist_plot(feature, bins, kde, log_scale, hue_var))

    st.sidebar.markdown("#### [Box Plots](#box-plots) ")

    st.write("### Box Plots")
    with st.expander("Box Plots"):
        box_feature = st.selectbox("Variable", options=feat_selections)
        box_group = st.selectbox(
            "Gruppierung",
            options=hue_options,
            key=widget_key,
            index=hue_options.index(None),
        )
        widget_key += 1
        box_color = st.selectbox(
            "Farbe",
            options=hue_options,
            key=widget_key,
            index=hue_options.index(None),
        )
        widget_key += 1
        st.pyplot(box_plot(box_feature, box_group, box_color))

    # Multivariate Plots
    st.sidebar.markdown("## Multivariate Plots")
    st.sidebar.markdown("#### [Scatter Plots](#scatter-plots) ")
    st.write("### Scatter Plots")
    with st.expander("Scatter Plots"):

        scatter_x = st.selectbox(
            "X-Achse", options=opt_selections, key=widget_key, index=0
        )
        widget_key += 1
        scatter_y = st.selectbox(
            "Y-Achse", options=opt_selections, key=widget_key, index=1
        )

        widget_key += 1
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

        st.pyplot(
            scatter_plot(scatter_x, scatter_y, scatter_color, scatter_size, alpha)
        )

    st.sidebar.markdown("#### [Dichte Plots](#scatter-plots) ")
    st.write("### Dichte Plots")
    with st.expander("Dichte Plots"):

        kde_x = st.selectbox("X-Achse", options=opt_selections, key=widget_key, index=0)
        widget_key += 1
        kde_y = st.selectbox("Y-Achse", options=opt_selections, key=widget_key, index=1)

        widget_key += 1
        kde_color = st.selectbox(
            "Farbe",
            options=opt_selections,
            key=widget_key,
            index=opt_selections.index(None),
        )
        widget_key += 1
        kde_fill = st.checkbox(
            "Fill",
            value=False,
            key=widget_key,
        )
        widget_key += 1
        st.pyplot(kde_plot(kde_x, kde_y, kde_color, kde_fill))
