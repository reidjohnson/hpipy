Accuracy Metrics
================

Accuracy metrics measure how well the index predicts actual property values. We will demonstrate how to calculate and visualize accuracy metrics.

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

Calculating Accuracy
--------------------

To calculate the accuracy of the index, you can use the ``accuracy`` function:

.. code-block:: python

    >>> from hpipy.utils.metrics import accuracy
    >>> from hpipy.utils.plotting import plot_index_accuracy

    >>> acc = accuracy(hpi)
    >>> acc.round(5).head()
       pair_id  rt_price    pred_price    error  log_error  pred_period
    0        1    577200  555423.08556 -0.03773   -0.03846           75
    1        2    488737  458143.88008 -0.06260   -0.06464           63
    2        3    570000  582619.73327  0.02214    0.02190           77
    3        4    402500  416336.17537  0.03438    0.03380           63
    4        5    385000  429625.48835  0.11591    0.10967           69

    >>> plot_index_accuracy(acc).properties(
    ...     title={"text": "Accuracy Metrics", "anchor": "middle"}
    ... )
    alt.VConcatChart(...)

.. invisible-altair-plot::

    import altair as alt
    from hpipy.datasets import load_seattle_sales
    from hpipy.price_index import RepeatTransactionIndex
    from hpipy.utils.metrics import accuracy
    from hpipy.utils.plotting import plot_index_accuracy

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
    acc = accuracy(hpi)
    spec = (
        plot_index_accuracy(acc)
        .properties(title={"text": "Accuracy Metrics", "anchor": "middle"})
        .to_dict()
    )
    for vrow in spec.get("vconcat", []):
        for hchart in vrow.get("hconcat", []):
            hchart.pop("width", None)
            hchart.pop("height", None)
    chart = alt.Chart.from_dict(spec).configure_view(continuousHeight=150, continuousWidth=275)

Rolling Window Analysis
-----------------------

You can also analyze accuracy over different time windows:

.. code-block:: python

    >>> rolling_acc = accuracy(hpi, window=12)
    >>> rolling_acc.round(5).head()
       pair_id  rt_price    pred_price    error  log_error  pred_period
    0        1    577200  555423.08556 -0.03773   -0.03846           75
    1        2    488737  458143.88008 -0.06260   -0.06464           63
    2        3    570000  582619.73327  0.02214    0.02190           77
    3        4    402500  416336.17537  0.03438    0.03380           63
    4        5    385000  429625.48835  0.11591    0.10967           69

The `window` parameter specifies the number of periods to use for the rolling calculation.
