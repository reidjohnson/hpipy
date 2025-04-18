Comparative Analysis
====================

It is often useful to compare the performance of different index methods. We will demonstrate how to do this by comparing metrics and visualizing indices.

Basic Setup
-----------

First, we will load some sales data and create several different indices:

.. code-block:: python

    >>> from hpipy.datasets import load_seattle_sales
    >>> from hpipy.extensions import NeuralNetworkIndex, RandomForestIndex
    >>> from hpipy.price_index import HedonicIndex, RepeatTransactionIndex

    >>> df = load_seattle_sales()

    >>> index_kwargs = {
    ...     "trans_data": df,
    ...     "prop_id": "pinx",
    ...     "trans_id": "sale_id",
    ...     "price": "sale_price",
    ...     "date": "sale_date",
    ...     "periodicity": "M",
    ... }
    >>> rt_kwargs = {
    ...     **index_kwargs,
    ...     "estimator": "robust",
    ...     "log_dep": True,
    ... }
    >>> hed_kwargs = {
    ...     **index_kwargs,
    ...     "dep_var": "price",
    ...     "ind_var": ["tot_sf", "beds", "baths"],
    ... }

    >>> hpi_rt = RepeatTransactionIndex.create_index(**rt_kwargs)
    >>> hpi_hed = HedonicIndex.create_index(**hed_kwargs)
    >>> hpi_rf = RandomForestIndex.create_index(**hed_kwargs)
    >>> hpi_nn = NeuralNetworkIndex.create_index(**hed_kwargs, preprocess_geo=False)

    >>> indices = [hpi_rt, hpi_hed, hpi_rf, hpi_nn]

Comparing Metrics
-----------------

Compare metrics between different index methods:

.. code-block:: python

    >>> import pandas as pd
    >>> from hpipy.utils.metrics import volatility

    >>> df_mean_vol = pd.DataFrame(
    ...    {
    ...        "Index": [idx.__class__.__name__ for idx in indices],
    ...        "Volatility": [volatility(idx)["mean"].iloc[0] for idx in indices],
    ...    }
    ... )

    >>> df_mean_vol.sort_values(by="Volatility").round(3)
                        Index  Volatility
    3      NeuralNetworkIndex       0.009
    2       RandomForestIndex       0.014
    0  RepeatTransactionIndex       0.017
    1            HedonicIndex       0.023

Visualizing Indices
-------------------

Visualize different index methods:

.. code-block:: python

    >>> import altair as alt
    >>> from hpipy.utils.plotting import plot_index

    >>> alt.layer(
    ...     (
    ...         plot_index(hpi_rt)
    ...         .transform_calculate(method="'Repeat Sales'")
    ...         .encode(color=alt.Color("method:N", title="Method"))
    ...     ),
    ...     (
    ...         plot_index(hpi_hed)
    ...         .transform_calculate(method="'Hedonic Pricing'")
    ...         .encode(color=alt.Color("method:N", title="Method"))
    ...     ),
    ...     (
    ...         plot_index(hpi_rf)
    ...         .transform_calculate(method="'Random Forest'")
    ...         .encode(color=alt.Color("method:N", title="Method"))
    ...     ),
    ...     (
    ...         plot_index(hpi_nn)
    ...         .transform_calculate(method="'Neural Network'")
    ...         .encode(color=alt.Color("method:N", title="Method"))
    ...     ),
    ... ).properties(title="Price Index Method Comparison")
    alt.LayerChart(...)

.. invisible-altair-plot::

    import altair as alt
    from hpipy.datasets import load_seattle_sales
    from hpipy.extensions import NeuralNetworkIndex, RandomForestIndex
    from hpipy.price_index import HedonicIndex, RepeatTransactionIndex
    from hpipy.utils.plotting import plot_index

    df = load_seattle_sales()
    index_kwargs = {
        "trans_data": df,
        "prop_id": "pinx",
        "trans_id": "sale_id",
        "price": "sale_price",
        "date": "sale_date",
        "periodicity": "M",
    }
    rt_kwargs = {
        **index_kwargs,
        "estimator": "robust",
        "log_dep": True,
    }
    hed_kwargs = {
        **index_kwargs,
        "dep_var": "price",
        "ind_var": ["tot_sf", "beds", "baths"],
    }
    hpi_rt = RepeatTransactionIndex.create_index(**rt_kwargs)
    hpi_hed = HedonicIndex.create_index(**hed_kwargs)
    hpi_rf = RandomForestIndex.create_index(**hed_kwargs)
    hpi_nn = NeuralNetworkIndex.create_index(**hed_kwargs, preprocess_geo=False)
    chart = alt.layer(
        (
            plot_index(hpi_rt)
            .transform_calculate(method="'Repeat Sales'")
            .encode(color=alt.Color("method:N", title="Method"))
        ),
        (
            plot_index(hpi_hed)
            .transform_calculate(method="'Hedonic Pricing'")
            .encode(color=alt.Color("method:N", title="Method"))
        ),
        (
            plot_index(hpi_rf)
            .transform_calculate(method="'Random Forest'")
            .encode(color=alt.Color("method:N", title="Method"))
        ),
        (
            plot_index(hpi_nn)
            .transform_calculate(method="'Neural Network'")
            .encode(color=alt.Color("method:N", title="Method"))
        ),
    ).properties(title="Price Index Method Comparison", width=500)
