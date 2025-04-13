Random Forest Method
====================

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
    >>> from hpipy.extensions import RandomForestIndex
    >>> from hpipy.period_table import PeriodTable
    >>> from hpipy.trans_data import HedonicTransactionData

    # Load and prepare data.
    >>> df = pd.read_csv("data/ex_sales.csv", parse_dates=["sale_date"])
    
    # Create period table.
    >>> sales_hdata = PeriodTable(df).create_period_table(
    ...     "sale_date",
    ...     periodicity="monthly",
    ... )

    # Prepare hedonic data.
    >>> trans_data = HedonicTransactionData(sales_hdata).create_transactions(
    ...     price="sale_price",
    ...     trans_id="sale_id",
    ...     prop_id="pinx",
    ... )

Creating the Index
------------------

Create a Random Forest-based index:

.. code-block:: python

    # Create the index.
    >>> hpi = RandomForestIndex.create_index(
    ...     trans_data=trans_data,
    ...     trans_id="sale_id",
    ...     prop_id="pinx",
    ...     date="sale_date",
    ...     price="sale_price",
    ...     dep_var="price",
    ...     ind_var=["tot_sf", "beds", "baths"],
    ...     n_estimators=100,
    ...     min_samples_leaf=5,
    ...     max_features="sqrt",
    ...     smooth=True,
    ... )

Parameters
----------

Key Random Forest parameters:

n_estimators : int
    Number of trees in the forest (default: 100).

min_samples_leaf : int
    Minimum samples required at leaf nodes (default: 5).

max_features : str or int
    Number of features to consider for splits ("sqrt", "log2", or int).

smooth : bool
    Whether to apply smoothing to the final index.

bootstrap : bool
    Whether to use bootstrap samples (default: True).

Feature Importance
------------------

Analyze feature importance:

.. code-block:: python

    # Get feature importance.
    >>> importance = hpi.model.model_obj.feature_importances_

Evaluating the Index
--------------------

Evaluate the random forest index using various metrics:

.. code-block:: python

    >>> from hpipy.utils.metrics import volatility
    >>> from hpipy.utils.plotting import plot_index

    # Calculate metrics.
    >>> vol = volatility(hpi)

    # Visualize results.
    >>> plot_index(hpi)
    alt.Chart(...)
