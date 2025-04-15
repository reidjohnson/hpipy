Comparative Analysis
====================

Compare different index methods:

.. code-block:: python

    >>> import pandas as pd
    >>> from hpipy.datasets import load_ex_sales
    >>> from hpipy.extensions import NeuralNetworkIndex, RandomForestIndex
    >>> from hpipy.price_index import HedonicIndex, RepeatTransactionIndex
    >>> from hpipy.utils.metrics import accuracy, volatility

    >>> df = load_ex_sales()

    >>> index_kwargs = {
    ...    "trans_data": df,
    ...    "prop_id": "pinx",
    ...    "trans_id": "sale_id",
    ...    "price": "sale_price",
    ...    "date": "sale_date",
    ...    "periodicity": "M",
    ... }
    >>> rt_kwargs = {
    ...    **index_kwargs,
    ...    "estimator": "robust",
    ...    "log_dep": True,
    ... }
    >>> hed_kwargs = {
    ...    **index_kwargs,
    ...    "dep_var": "price",
    ...    "ind_var": ["tot_sf", "beds", "baths"],
    ... }

    >>> rt_hpi = RepeatTransactionIndex.create_index(**rt_kwargs)
    >>> hed_hpi = HedonicIndex.create_index(**hed_kwargs)
    >>> rf_hpi = RandomForestIndex.create_index(**hed_kwargs)
    >>> nn_hpi = NeuralNetworkIndex.create_index(**hed_kwargs)

    >>> indices = [rt_hpi, hed_hpi, rf_hpi, nn_hpi]

    >>> volatilities = [volatility(index) for index in indices]
    >>> volatilities[0].head()
           roll      mean    median
    1  0.016313  0.057788  0.051364
    2  0.022051  0.057788  0.051364
    3  0.026690  0.057788  0.051364
    4  0.016093  0.057788  0.051364
    5  0.024070  0.057788  0.051364

    >>> volatilities[1].head()
           roll      mean    median
    1  0.046572  0.059311  0.052778
    2  0.045874  0.059311  0.052778
    3  0.052778  0.059311  0.052778
    4  0.026319  0.059311  0.052778
    5  0.067284  0.059311  0.052778
