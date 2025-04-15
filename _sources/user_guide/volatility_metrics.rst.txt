Volatility Metrics
==================

Volatility metrics measure the stability and smoothness of the index:

.. code-block:: python

    >>> import pandas as pd
    >>> from hpipy.datasets import load_seattle_sales
    >>> from hpipy.price_index import RepeatTransactionIndex
    >>> from hpipy.utils.metrics import volatility
    >>> from hpipy.utils.plotting import plot_index_volatility

    >>> df = load_seattle_sales()

    >>> hpi = RepeatTransactionIndex.create_index(
    ...    trans_data=df,
    ...    prop_id="pinx",
    ...    trans_id="sale_id",
    ...    price="sale_price",
    ...    date="sale_date",
    ...    periodicity="M",
    ...    estimator="robust",
    ...    log_dep=True,
    ...    smooth=True,
    ... )

    >>> vol = volatility(hpi)
    >>> vol.head()
           roll      mean    median
    1  0.024743  0.017214  0.016522
    2  0.027512  0.017214  0.016522
    3  0.023363  0.017214  0.016522
    4  0.015850  0.017214  0.016522
    5  0.004755  0.017214  0.016522

    >>> plot_index_volatility(vol).properties(title="Volatility Metrics")
    alt.LayerChart(...)

.. invisible-altair-plot::

    import pandas as pd
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
