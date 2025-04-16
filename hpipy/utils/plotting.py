"""Plotting utilities."""

import copy

import altair as alt
import numpy as np
import pandas as pd

from hpipy.price_index import BaseHousePriceIndex


def plot_index(
    hpi: BaseHousePriceIndex,
    show_imputed: bool = False,
    smooth: bool = False,
) -> alt.Chart:
    """Plot index."""
    hpi = copy.deepcopy(hpi)

    dates = (
        hpi.model.period_table[["period", "name"]]
        .merge(hpi.periods, on="period", how="right")
        .assign(date=lambda x: pd.to_datetime(x["name"], format="%Y-%b"))
        .assign(date=lambda x: x["date"].dt.to_period("M").dt.to_timestamp("M"))
        .loc[:, "date"]
    )
    values = hpi.value
    imputed = hpi.imputed

    source = pd.DataFrame({"x": dates, "y": values})

    base = (
        alt.Chart(source)
        .mark_line(size=4)
        .encode(
            x=alt.X(
                "x:T",
                axis=alt.Axis(
                    format="%Y", labelAngle=0, title="Time Period", values=dates[::12].values
                ),
            ),
            y=alt.Y("y:Q", scale=alt.Scale(zero=False), title="Index Value"),
            tooltip=[
                alt.Tooltip("x:T", title="Time Period"),
                alt.Tooltip("y:Q", format=".3f", title="Index Value"),
            ],
        )
    )

    if show_imputed:
        source = pd.DataFrame({"x": dates, "y": values, "imp": imputed})

        circles = (
            alt.Chart(source[source["imp"] == 1])
            .transform_calculate(imp="if(datum.imp == 1, 'True', 'False')")
            .mark_circle(color="red", size=250)
            .encode(
                x=alt.X("x:T", axis=alt.Axis(format="%Y", labelAngle=0, title="Time Period")),
                y=alt.Y("y:Q", scale=alt.Scale(zero=False), title="Index Value"),
                tooltip=[
                    alt.Tooltip("x:T", title="Time Period"),
                    alt.Tooltip("y:Q", format=".3f", title="Index Value"),
                    alt.Tooltip("imp:N", title="Imputed"),
                ],
            )
        )
        base += circles

    if smooth:
        smooth = hpi.smooth

        source = pd.DataFrame({"x": dates, "y": smooth, "imp": imputed})

        smooth = (
            alt.Chart(source)
            .transform_calculate(imp="if(datum.imp == 1, 'True', 'False')")
            .mark_line(color="red", size=3)
            .encode(
                x=alt.X("x:T", axis=alt.Axis(format="%Y", labelAngle=0, title="Time Period")),
                y=alt.Y("y:Q", scale=alt.Scale(zero=False), title="Index Value"),
                tooltip=[
                    alt.Tooltip("x:T", title="Time Period"),
                    alt.Tooltip("y:Q", format=".3f", title="Index Value"),
                    alt.Tooltip("imp:N", title="Imputed"),
                ],
            )
        )
        base += smooth

    chart = base.properties(height=300, width=800)

    return chart


def plot_index_accuracy(error_df: pd.DataFrame, size: int = 3) -> alt.Chart:
    """Plot index accuracy."""
    source = error_df.copy()

    bar1 = (
        alt.Chart(source)
        .mark_boxplot(clip=True, extent="min-max", size=size)
        .transform_calculate(abserror=np.abs(alt.datum["error"]))
        .encode(
            x=alt.X(
                "pred_period:Q",
                axis=alt.Axis(tickMinStep=1),
                scale=alt.Scale(zero=False),
                title="Time Period",
            ),
            y=alt.Y(
                "abserror:Q",
                scale=alt.Scale(domain=[0, source["error"].quantile(0.99)]),
                title="Absolute Error",
            ),
            tooltip=[
                alt.Tooltip("pred_period:Q", title="Time Period"),
                alt.Tooltip("abserror:Q", format=".3f", title="Absolute Error"),
            ],
        )
        .properties(height=150, width=375)
    )

    bar2 = (
        alt.Chart(source)
        .mark_boxplot(clip=True, extent="min-max", size=size)
        .encode(
            x=alt.X(
                "pred_period:Q",
                axis=alt.Axis(tickMinStep=1),
                scale=alt.Scale(zero=False),
                title="Time Period",
            ),
            y=alt.Y(
                "error:Q",
                scale=alt.Scale(domain=source["error"].quantile([0.01, 0.99]).values),
                title="Error",
            ),
            tooltip=[
                alt.Tooltip("pred_period:Q", title="Time Period"),
                alt.Tooltip("error:Q", format=".3f", title="Error"),
            ],
        )
        .properties(height=150, width=375)
    )

    density1 = (
        alt.Chart(source)
        .mark_area(clip=True)
        .transform_calculate(abserror=np.abs(alt.datum["error"]))
        .transform_density("abserror", as_=["abserror", "density"])
        .encode(
            x=alt.X(
                "abserror:Q",
                scale=alt.Scale(domain=[0, source["error"].quantile(0.99)]),
                title="Absolute Error",
            ),
            y=alt.Y("density:Q", title="Density of Error"),
            tooltip=[
                alt.Tooltip("abserror:Q", format=".3f", title="Absolute Error"),
                alt.Tooltip("density:Q", format=".3f", title="Density of Error"),
            ],
        )
        .properties(height=150, width=375)
    )

    density2 = (
        alt.Chart(source)
        .mark_area(clip=True)
        .transform_density("error", as_=["error", "density"])
        .encode(
            x=alt.X(
                "error:Q",
                scale=alt.Scale(domain=source["error"].quantile([0.01, 0.99]).values),
                title="Error",
            ),
            y=alt.Y("density:Q", title="Density of Error"),
            tooltip=[
                alt.Tooltip("error:Q", format=".3f", title="Error"),
                alt.Tooltip("density:Q", format=".3f", title="Density of Error"),
            ],
        )
        .properties(height=150, width=375)
    )

    chart = (bar1 | bar2) & (density1 | density2)

    return chart


def plot_index_volatility(volatility_df: pd.DataFrame) -> alt.Chart:
    """Plot index volatility."""
    source = pd.DataFrame(
        {
            "x": volatility_df.index,
            "y": volatility_df["roll"],
            "mean": volatility_df["mean"],
            "median": volatility_df["median"],
        }
    )

    base = (
        alt.Chart(source)
        .mark_line(size=4)
        .encode(
            x=alt.X("x:Q", title="Time Period"),
            y=alt.Y("y:Q", scale=alt.Scale(zero=False), title="Volatility"),
            tooltip=[
                alt.Tooltip("x:Q", title="Time Period"),
                alt.Tooltip("y:Q", format=".3f", title="Volatility"),
            ],
        )
    )

    mean = base.mark_line(color="gray", size=3, strokeDash=[4, 4]).encode(y=alt.Y("mean:Q"))
    median = base.mark_line(color="gray", size=3, strokeDash=[2, 4]).encode(y=alt.Y("median:Q"))

    chart = (base + mean + median).properties(height=300, width=800)

    return chart


def plot_series_volatility(hpi_series: BaseHousePriceIndex, smooth: bool = False) -> alt.Chart:
    """Plot volatility for a series of indices."""
    index = [pd.Series([idx] * len(x.periods)) for idx, x in enumerate(hpi_series.hpis)]
    periods = [x.periods for x in hpi_series.hpis]
    values = [(x.smooth if smooth else x.value) for x in hpi_series.hpis]

    source = pd.DataFrame(
        {
            "index": pd.concat(index),
            "period": pd.concat(periods),
            "value": pd.concat(values),
        }
    )

    base = (
        alt.Chart()
        .mark_line(size=4)
        .encode(
            x=alt.X("period:Q", title="Time Period"),
            y=alt.Y("value:Q", scale=alt.Scale(zero=False), title="Volatility"),
            tooltip=[
                alt.Tooltip("period:Q", title="Time Period"),
                alt.Tooltip("value:Q", format=".3f", title="Volatility"),
            ],
        )
        .properties(height=300, width=800)
    )

    chart1 = (
        base.mark_line(size=2)
        .encode(color=alt.Color("index:N", legend=None, scale=alt.Scale(scheme="greys")))
        .properties(data=source[source["index"] < source["index"].max()])
    )
    chart2 = (
        base.mark_line(color="red", size=4)
        .encode()
        .properties(data=source[source["index"] == source["index"].max()])
    )
    chart = (chart1 + chart2).resolve_scale(color="independent")

    return chart


def plot_series_revision(
    hpi_series: BaseHousePriceIndex,
    measure: str = "median",
    smooth: bool = False,
) -> alt.LayerChart:
    """Plot revision for a series of indices."""
    source = hpi_series.revision_smooth if smooth else hpi_series.revision

    measure_title = "Mean" if measure == "mean" else "Median"

    base = (
        alt.Chart()
        .mark_bar(size=20)
        .encode(
            x=alt.X("period:Q", title="Time Period"),
            y=alt.Y(
                f"{measure}:Q",
                title=f"{measure_title} Revision",
            ),
            color=alt.condition(
                alt.datum[measure] > 0,
                alt.value("steelblue"),
                alt.value("orange"),
            ),
            tooltip=[
                alt.Tooltip("period:Q", title="Time Period"),
                alt.Tooltip(f"{measure}:Q", format=".3f", title=f"{measure_title} Revision"),
            ],
        )
        .properties(height=300, width=800)
    )

    line = (
        alt.Chart()
        .mark_rule(color="gray", size=3, strokeDash=[4, 4])
        .encode(y=alt.Y(f"mean({measure}):Q"))
    )

    chart = alt.layer(base, line, data=source)

    return chart
