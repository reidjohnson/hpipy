Interpretable Random Forest
===========================

The random forest method applies an ensemble machine learning approach to index construction, automatically capturing nonlinear relationships and complex interactions between property features. It does not require prior assumptions about functional form and can handle large datasets effectively. With sufficient data, it offers a flexible and robust alternative to traditional models.

.. note::

    Background on model construction for the interpretable random forest can be found in:

    Krause (2019), "A Machine Learning Approach to House Prince Indexes". `https://github.com/andykrause/hpi_research/tree/master/papers/irf <https://github.com/andykrause/hpi_research/tree/master/papers/irf>`_.

Overview
--------

The random forest method:

1. Uses an ensemble of decision trees
2. Handles both numerical and categorical features
3. Automatically handles non-linear relationships
4. Uses partial dependence plots to derive index

Data Preparation
----------------

Similar to the hedonic method, you need:

* A date column (e.g., "sale_date")
* A price column (e.g., "sale_price")
* Property characteristics
* A transaction identifier

Example setup:

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

Create a random forest-based index:

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

The main parameters for random forest index creation are:

.. admonition:: Parameters
   :class: hint

   **dep_var** : str
       Dependent variable to model.

   **ind_var** : list
       Independent variables to use in the model.

   **log_dep** : bool
       Whether to use log of price as dependent variable (recommended).

   **n_estimators** : int
       Number of trees in the forest (default: 100).

Feature Importance
------------------

The random forest model is implemented using the `RandomForestRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_ class from scikit-learn. We can access the feature importance from the model object:

.. code-block:: python

    >>> importance = hpi.model.model_obj.feature_importances_
    >>> importance
    array(...)

Similarly, we can leverage the `partial_dependence <https://scikit-learn.org/stable/modules/partial_dependence.html>`_ function to plot the partial dependence of the model on particular features. Here, we plot the partial dependence of the model on the transaction period, which is used to derive the index:

.. code-block:: python

    >>> import altair as alt
    >>> import pandas as pd
    >>> from sklearn.inspection import partial_dependence

    >>> predictions = partial_dependence(
    ...    hpi.model.model_obj,
    ...    hpi.model.X,
    ...    features=["trans_period"],
    ... )
    >>> df_pdp = pd.DataFrame({k: v[0] for k, v in predictions.items()})

    >>> alt.Chart(df_pdp).mark_line(size=4).encode(
    ...    x=alt.X("grid_values:Q", title="Transaction Period"),
    ...    y=alt.Y(
    ...        "average:Q", scale=alt.Scale(zero=False), title="Partial Dependence"
    ...    ),
    ...    tooltip=[
    ...        alt.Tooltip("grid_values", title="Transaction Period"),
    ...        alt.Tooltip("average", format=".3f", title="Partial Dependence"),
    ...    ],
    ... ).properties(width=600)
    alt.Chart(...)

.. invisible-altair-plot::

    import altair as alt
    import pandas as pd
    from sklearn.inspection import partial_dependence
    from hpipy.datasets import load_ex_sales
    from hpipy.extensions import RandomForestIndex
    from hpipy.period_table import PeriodTable
    from hpipy.trans_data import HedonicTransactionData

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
        log_dep=True,
        n_estimators=100,
        smooth=True,
        random_seed=0,
    )
    predictions = partial_dependence(hpi.model.model_obj, hpi.model.X, features=["trans_period"])
    df_pdp = pd.DataFrame({k: v[0] for k, v in predictions.items()})
    chart = (
        alt.Chart(df_pdp)
        .mark_line(size=4)
        .encode(
            x=alt.X("grid_values", title="Transaction Period"),
            y=alt.Y("average", scale=alt.Scale(zero=False), title="Partial Dependence"),
            tooltip=[
                alt.Tooltip("grid_values", title="Transaction Period"),
                alt.Tooltip("average", format=".3f", title="Partial Dependence"),
            ],
        )
        .properties(width=600)
    )

Evaluating the Index
--------------------

Evaluate the random forest index using various metrics:

.. code-block:: python

    >>> from hpipy.utils.metrics import volatility
    >>> from hpipy.utils.plotting import plot_index

    # Calculate metrics.
    >>> vol = volatility(hpi)

    # Visualize the index.
    >>> plot_index(hpi, smooth=True).properties(title="Random Forest Index")
    alt.LayerChart(...)

.. invisible-altair-plot::

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
