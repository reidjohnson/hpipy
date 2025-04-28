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
    >>> vol.round(5).head()
          roll     mean   median
    1  0.02474  0.01721  0.01652
    2  0.02751  0.01721  0.01652
    3  0.02336  0.01721  0.01652
    4  0.01585  0.01721  0.01652
    5  0.00476  0.01721  0.01652

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
    >>> rolling_vol.round(5).head()
           roll     mean   median
    6   0.01655  0.01799  0.01739
    7   0.02159  0.01799  0.01739
    8   0.02078  0.01799  0.01739
    9   0.01927  0.01799  0.01739
    10  0.01837  0.01799  0.01739

The `window` parameter specifies the number of periods to use for the rolling calculation.
