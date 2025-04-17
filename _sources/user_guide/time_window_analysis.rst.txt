Time Window Analysis
====================

Analyze metrics over different time windows:

.. code-block:: python

    >>> from hpipy.datasets import load_seattle_sales
    >>> from hpipy.price_index import RepeatTransactionIndex
    >>> from hpipy.utils.metrics import accuracy, volatility

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

    >>> rolling_acc = accuracy(hpi, window=12)
    >>> rolling_acc.head()
       pair_id  rt_price     pred_price     error  log_error  pred_period
    0        1    577200  555423.085557 -0.037729  -0.038459           75
    1        2    488737  458143.880078 -0.062596  -0.064641           63
    2        3    570000  582619.733273  0.022140   0.021898           77
    3        4    402500  416336.175366  0.034376   0.033798           63
    4        5    385000  429625.488353  0.115910   0.109671           69

    >>> rolling_vol = volatility(hpi, window=12)
    >>> rolling_vol.head()
            roll      mean    median
    6   0.016548  0.017985  0.017387
    7   0.021586  0.017985  0.017387
    8   0.020780  0.017985  0.017387
    9   0.019267  0.017985  0.017387
    10  0.018374  0.017985  0.017387
