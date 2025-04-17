Accuracy Metrics
================

Accuracy metrics measure how well the index predicts actual property values:

.. code-block:: python

    >>> from hpipy.datasets import load_seattle_sales
    >>> from hpipy.price_index import RepeatTransactionIndex
    >>> from hpipy.utils.metrics import accuracy
    >>> from hpipy.utils.plotting import plot_index_accuracy

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

    >>> acc = accuracy(hpi)
    >>> acc.head()
       pair_id  rt_price     pred_price     error  log_error  pred_period
    0        1    577200  555423.085557 -0.037729  -0.038459           75
    1        2    488737  458143.880078 -0.062596  -0.064641           63
    2        3    570000  582619.733273  0.022140   0.021898           77
    3        4    402500  416336.175366  0.034376   0.033798           63
    4        5    385000  429625.488353  0.115910   0.109671           69

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
