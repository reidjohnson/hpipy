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
    0        1    577200  555422.458246 -0.037730  -0.038460           75
    1        2    488737  458143.826186 -0.062596  -0.064641           63
    2        3    570000  582619.340460  0.022139   0.021898           77
    3        4    402500  416336.126392  0.034375   0.033798           63
    4        5    385000  429625.133353  0.115909   0.109669           69

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
