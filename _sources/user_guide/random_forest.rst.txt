Random Forest
=============

The Random Forest method provides a machine learning approach to house price index creation, offering advantages like handling non-linear relationships and complex interactions between features automatically.

Overview
--------

The Random Forest approach in hpiPy:

1. Uses an ensemble of decision trees
2. Handles both numerical and categorical features
3. Includes partial dependence plots
4. Automatically handles non-linear relationships

Data Preparation
----------------

Similar to the hedonic method, you need:

* A date column (e.g., "sale_date")
* A price column (e.g., "sale_price")
* Property characteristics
* A transaction identifier

Example setup:

.. code-block:: python

    >>> import pandas as pd
    >>> from hpipy.datasets import load_ex_sales
    >>> from hpipy.period_table import PeriodTable
    >>> from hpipy.trans_data import HedonicTransactionData

    # Load and prepare data.
    >>> df = load_ex_sales()
    
    # Create period table.
    >>> sales_hdata = PeriodTable(df).create_period_table(
    ...     "sale_date",
    ...     periodicity="monthly",
    ... )

    # Prepare hedonic data.
    >>> trans_data = HedonicTransactionData(sales_hdata).create_transactions(
    ...     prop_id="pinx",
    ...     trans_id="sale_id",
    ...     price="sale_price",
    ... )

Creating the Index
------------------

Create a Random Forest-based index:

.. code-block:: python

    >>> from hpipy.extensions import RandomForestIndex

    # Create the index.
    >>> hpi = RandomForestIndex.create_index(
    ...     trans_data=trans_data,
    ...     prop_id="pinx",
    ...     trans_id="sale_id",
    ...     price="sale_price",
    ...     date="sale_date",
    ...     dep_var="price",
    ...     ind_var=["tot_sf", "beds", "baths"],
    ...     log_dep=True,
    ...     n_estimators=100,
    ...     smooth=True,
    ...     random_seed=0,
    ... )

Parameters
----------

Key Random Forest parameters:

dep_var : str
    Dependent variable to model.

ind_var : list
    Independent variables to use in the model.

log_dep : bool
    Whether to use log of price as dependent variable (recommended).

n_estimators : int
    Number of trees in the forest (default: 100).

Feature Importance
------------------

Analyze feature importance:

.. code-block:: python

    >>> importance = hpi.model.model_obj.feature_importances_
    >>> importance
    array(...)

Evaluating the Index
--------------------

Evaluate the random forest index using various metrics:

.. code-block:: python

    >>> from hpipy.utils.metrics import volatility
    >>> from hpipy.utils.plotting import plot_index

    # Calculate metrics.
    >>> vol = volatility(hpi)

    # Visualize results.
    >>> plot_index(hpi, smooth=True).properties(title="Random Forest Index")
    alt.LayerChart(...)

.. invisible-altair-plot::

    import pandas as pd
    from hpipy.datasets import load_ex_sales
    from hpipy.extensions import RandomForestIndex
    from hpipy.period_table import PeriodTable
    from hpipy.trans_data import HedonicTransactionData
    from hpipy.utils.plotting import plot_index
    df = load_ex_sales()
    sales_hdata = PeriodTable(df).create_period_table("sale_date", periodicity="monthly")
    trans_data = HedonicTransactionData(sales_hdata).create_transactions(
        prop_id="pinx", trans_id="sale_id", price="sale_price"
    )
    hpi = RandomForestIndex.create_index(
        trans_data=trans_data,
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        dep_var="price",
        ind_var=["tot_sf", "beds", "baths"],
        n_estimators=100,
        min_samples_leaf=5,
        max_features="sqrt",
        smooth=True,
        random_seed=0,
    )
    chart = plot_index(hpi, smooth=True).properties(title="Random Forest Index", width=600)
