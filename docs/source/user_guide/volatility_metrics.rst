Volatility Metrics
==================

Volatility metrics measure the stability and smoothness of the index. We will demonstrate how to calculate and visualize volatility metrics.

Basic Setup
-----------

First, we will import the necessary modules, load some sales data, and create an index:

.. code-block:: python

    >>> from hpipy.datasets import load_seattle_sales
    >>> from hpipy.price_index import RepeatTransactionIndex

    >>> df = load_seattle_sales()

    >>> hpi = RepeatTransactionIndex.create_index(
    ...     trans_data=df,
    ...     prop_id="pinx",
    ...     trans_id="sale_id",
    ...     price="sale_price",
    ...     date="sale_date",
    ...     periodicity="M",
    ...     estimator="robust",
    ...     log_dep=True,
    ...     smooth=True,
    ... )

Calculating Volatility
----------------------

To calculate the volatility of the index, you can use the ``volatility`` function:

.. code-block:: python

    >>> from hpipy.utils.metrics import volatility
    >>> from hpipy.utils.plotting import plot_index_volatility

    >>> vol = volatility(hpi)
    >>> vol.round(3).head()
        roll   mean  median
    1  0.025  0.017   0.017
    2  0.028  0.017   0.017
    3  0.023  0.017   0.017
    4  0.016  0.017   0.017
    5  0.005  0.017   0.017

    >>> plot_index_volatility(vol).properties(title="Volatility Metrics")
    alt.LayerChart(...)

.. invisible-altair-plot::

    from hpipy.datasets import load_seattle_sales
    from hpipy.price_index import RepeatTransactionIndex
    from hpipy.utils.metrics import volatility
    from hpipy.utils.plotting import plot_index_volatility

    df = load_seattle_sales()
    hpi = RepeatTransactionIndex.create_index(
        trans_data=df,
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="M",
        estimator="robust",
        log_dep=True,
        smooth=True,
    )
    vol = volatility(hpi)
    chart = plot_index_volatility(vol).properties(title="Volatility Metrics", width=600)

Rolling Window Analysis
-----------------------

You can also analyze volatility over different time windows:

.. code-block:: python

    >>> rolling_vol = volatility(hpi, window=12)
    >>> rolling_vol.round(3).head()
         roll   mean  median
    6   0.017  0.018   0.017
    7   0.022  0.018   0.017
    8   0.021  0.018   0.017
    9   0.019  0.018   0.017
    10  0.018  0.018   0.017

The `window` parameter specifies the number of periods to use for the rolling calculation.
