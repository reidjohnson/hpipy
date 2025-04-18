Series Metrics
==============

We will demonstrate how to analyze and visualize house price index series. We will cover accuracy analysis, volatility measurements, revision tracking, and visualization techniques.

Basic Setup
-----------

First, we will import the necessary modules and load some sales data:

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

Now, create and visualize a series of indices from the index:

.. code-block:: python

    >>> from hpipy.utils.plotting import plot_series

    >>> hpi_series = hpi.create_series(train_period=24, max_period=30, smooth=True)

    # Plot index series.
    >>> plot_series(hpi_series)
    alt.LayerChart(...)

.. invisible-altair-plot::

    from hpipy.datasets import load_seattle_sales
    from hpipy.price_index import RepeatTransactionIndex
    from hpipy.utils.plotting import plot_series

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
    hpi_series = hpi.create_series(train_period=24, max_period=30, smooth=True)
    chart = plot_series(hpi_series).properties(width=600)

Accuracy Analysis
-----------------

Evaluate how well an index predicts actual property values using the ``series_accuracy`` function:

.. code-block:: python

    >>> from hpipy.utils.metrics import series_accuracy
    >>> from hpipy.utils.plotting import plot_series_accuracy

    # Calculate accuracy metrics.
    >>> df_accuracy = series_accuracy(hpi_series)
    >>> df_accuracy.round(3).head()
       index  pair_id  rt_price  pred_price  error  log_error  pred_period
    0      0      606    899000  703636.692 -0.217     -0.245            2
    1      0     2644    330000  336918.618  0.021      0.021            3
    2      0     3634    535000  549751.551  0.028      0.027            4
    3      0      387    549900  582136.997  0.059      0.057            5
    4      0     3269    305000  251097.865 -0.177     -0.194            5

    # Plot accuracy over time.
    >>> plot_series_accuracy(df_accuracy)
    alt.LayerChart(...)

.. invisible-altair-plot::

    from hpipy.datasets import load_seattle_sales
    from hpipy.price_index import RepeatTransactionIndex
    from hpipy.utils.metrics import series_accuracy
    from hpipy.utils.plotting import plot_series_accuracy

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
    hpi_series = hpi.create_series(train_period=24, max_period=30, smooth=True)
    df_accuracy = series_accuracy(hpi_series)
    chart = plot_series_accuracy(df_accuracy).properties(width=600)

Volatility Analysis
-------------------

Measure index volatility using the ``series_volatility`` function:

.. code-block:: python

    >>> from hpipy.utils.metrics import series_volatility
    >>> from hpipy.utils.plotting import plot_series_volatility

    # Calculate volatility metrics.
    >>> df_volatility = series_volatility(hpi_series)
    >>> df_volatility.round(3).head()
       index  period   roll   mean  median
    0      0       1  0.233  0.215    0.14
    1      0       2  0.119  0.215    0.14
    2      0       3  0.053  0.215    0.14
    3      0       4  0.047  0.215    0.14
    4      0       5  0.100  0.215    0.14

    # Plot volatility over time.
    >>> plot_series_volatility(df_volatility)
    alt.LayerChart(...)

.. invisible-altair-plot::

    from hpipy.datasets import load_seattle_sales
    from hpipy.price_index import RepeatTransactionIndex
    from hpipy.utils.metrics import series_volatility
    from hpipy.utils.plotting import plot_series_volatility

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
    hpi_series = hpi.create_series(train_period=24, max_period=30, smooth=True)
    df_volatility = series_volatility(hpi_series)
    chart = plot_series_volatility(df_volatility).properties(width=600)

Revision Analysis
-----------------

Track how index values change as new data becomes available using the ``revision`` function:

.. code-block:: python

    >>> from hpipy.utils.metrics import revision
    >>> from hpipy.utils.plotting import plot_series_revision

    # Calculate revision metrics.
    >>> df_revision = revision(hpi_series)
    >>> df_revision.round(3).head()
       period   mean  median
    0       1  0.000   0.000
    1       2 -0.161  -0.243
    2       3 -1.108   0.037
    3       4 -2.159  -1.220
    4       5 -1.570  -1.097

    # Plot revision analysis.
    >>> plot_series_revision(df_revision)
    alt.LayerChart(...)

.. invisible-altair-plot::

    from hpipy.datasets import load_seattle_sales
    from hpipy.price_index import RepeatTransactionIndex
    from hpipy.utils.metrics import revision
    from hpipy.utils.plotting import plot_series_revision

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
    hpi_series = hpi.create_series(train_period=24, max_period=30, smooth=True)
    df_revision = revision(hpi_series)
    chart = plot_series_revision(df_revision).properties(width=600)

See Also
--------

- :doc:`accuracy_metrics` for information on accuracy metrics.
- :doc:`volatility_metrics` for coverage of volatility metrics.
- :doc:`revision_metrics` for a guide to revision metrics.
