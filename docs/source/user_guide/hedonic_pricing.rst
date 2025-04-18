Hedonic Pricing
===============

The hedonic pricing method constructs house price indices by modeling sale prices as a function of observable property characteristics (such as size, location, and age) and time. This approach uses all available transactions and can produce robust indices when the model is well-specified. It offers transparency and flexibility, making it a common baseline method for house price analysis.

.. note::

    Background on basic model construction for hedonic pricing models can be found in:

    Bourassa et al. (2006), "A Simple Alternative House Price Index Method",
    *Journal of Housing Economics*, 15(1), 80-97. DOI: `10.1016/j.jhe.2006.03.001 <https://doi.org/10.1016/j.jhe.2006.03.001>`_.

Data Preparation
----------------

For hedonic pricing, your data should include:

* A date column (e.g., "sale_date")
* A price column (e.g., "sale_price")
* Property characteristics columns (e.g., "sqft", "bedrooms", "bathrooms")
* A transaction identifier column (e.g., "sale_id")

You can prepare your data as follows:

.. code-block:: python

    >>> from hpipy.datasets import load_ex_sales
    >>> from hpipy.period_table import PeriodTable
    >>> from hpipy.trans_data import HedonicTransactionData

    # Load sales data.
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

Once your data is prepared, you can create the hedonic price index:

.. code-block:: python

    >>> from hpipy.price_index import HedonicIndex

    # Create the index.
    >>> hpi = HedonicIndex.create_index(
    ...     trans_data=trans_data,
    ...     price="sale_price",
    ...     date="sale_date",
    ...     dep_var="price",
    ...     ind_var=["tot_sf", "beds", "baths"],
    ...     estimator="base",  # or "robust", "weighted"
    ...     log_dep=True,
    ...     smooth=True,
    ... )

Parameters
----------

The main parameters for hedonic index creation are:

.. admonition:: Parameters
   :class: hint

   **estimator** : str
       The type of estimator to use:

       * "base": Standard OLS estimation
       * "robust": Robust regression (less sensitive to outliers)
       * "weighted": Weighted regression

   **dep_var** : str
       Dependent variable to model.

   **ind_var** : list
       Independent variables to use in the model.

   **log_dep** : bool
       Whether to use log of price as dependent variable (recommended).

Advanced Usage
--------------

For more control over the hedonic model:

.. code-block:: python

    >>> from hpipy.price_index import HedonicIndex
    >>> from hpipy.price_model import HedonicModel

    # Create and fit the model.
    >>> model = HedonicModel(trans_data).fit(
    ...     dep_var="price",
    ...     ind_var=["tot_sf", "beds", "baths"],
    ...     estimator="base",
    ...     log_dep=True,
    ... )

    # Create the index.
    >>> hpi = HedonicIndex.from_model(model, trans_data=trans_data, smooth=True)

Feature Engineering
-------------------

The hedonic method often benefits from careful feature engineering:

1. Numeric Transformations:

   .. code-block:: python

       >>> import numpy as np

       # Log transform skewed features.
       >>> df["log_sqft"] = np.log(df["tot_sf"])

       # Create interaction terms.
       >>> df["price_per_sqft"] = df["sale_price"] / df["tot_sf"]

2. Categorical Features:

   .. code-block:: python

       >>> import pandas as pd

       >>> cat_cols = ["use_type", "area"]

       # One-hot encode categorical variables.
       >>> df = pd.get_dummies(df, columns=cat_cols)

3. Spatial Features:

   .. code-block:: python

       >>> lat_col, lon_col = "latitude", "longitude"

       # Create location-based features.
       >>> df["lat_lon"] = (
       ...     df.loc[:, [lat_col, lon_col]]
       ...     .round(2)
       ...     .astype(str)
       ...     .agg("_".join, axis=1)
       ... )

Evaluating the Index
--------------------

Evaluate the hedonic index using various metrics:

.. code-block:: python

    >>> from hpipy.utils.metrics import volatility
    >>> from hpipy.utils.plotting import plot_index

    # Calculate metrics.
    >>> vol = volatility(hpi)

    # Visualize the index.
    >>> plot_index(hpi, smooth=True).properties(title="Hedonic Index")
    alt.LayerChart(...)

.. invisible-altair-plot::

    from hpipy.datasets import load_ex_sales
    from hpipy.period_table import PeriodTable
    from hpipy.price_index import HedonicIndex
    from hpipy.price_model import HedonicModel
    from hpipy.trans_data import HedonicTransactionData
    from hpipy.utils.plotting import plot_index

    df = load_ex_sales()
    sales_hdata = PeriodTable(df).create_period_table("sale_date", periodicity="monthly")
    trans_data = HedonicTransactionData(sales_hdata).create_transactions(
        prop_id="pinx", trans_id="sale_id", price="sale_price"
    )
    model = HedonicModel(trans_data).fit(
        dep_var="price", ind_var=["tot_sf", "beds", "baths"], estimator="base", log_dep=True
    )
    hpi = HedonicIndex.from_model(model, trans_data=trans_data, smooth=True)
    chart = plot_index(hpi, smooth=True).properties(title="Hedonic Index", width=600)
