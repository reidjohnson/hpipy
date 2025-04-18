Repeat Sales
============

The repeat sales method estimates house price indices by analyzing changes in the sale price of the same property over time. By comparing repeated transactions of individual homes, it controls for unobserved property characteristics. It is widely used in practice and offers a simple and interpretable approach to index construction. However, it relies on a limited subset of the data and may be less effective in markets with low turnover or significant property changes between sales.

.. note::

    Background on basic model construction for repeat sales models can be found in:

    Case and Quigley (1991), "The Dynamics of Real Estate Prices", *The Review of Economics and Statistics*, 73(1), 50-58. DOI: `10.2307/2109686 <https://doi.org/10.2307/2109686>`_.

Data Preparation
----------------

To create a repeat sales index, you first need to prepare your data. The data should be in a format where each row represents a property sale with the following columns:

* A date column (e.g., "sale_date")
* A price column (e.g., "sale_price")
* A property identifier column (e.g., "pinx")
* A transaction identifier column (e.g., "sale_id")

You can prepare your data as follows:

.. code-block:: python

    >>> from hpipy.datasets import load_ex_sales
    >>> from hpipy.period_table import PeriodTable
    >>> from hpipy.trans_data import RepeatTransactionData

    # Load sales data.
    >>> df = load_ex_sales()

    # Create period table.
    >>> sales_hdata = PeriodTable(df).create_period_table(
    ...     "sale_date",
    ...     periodicity="monthly",  # or "quarterly", "yearly"
    ... )

    # Create repeat sales pairs.
    >>> trans_data = RepeatTransactionData(sales_hdata).create_transactions(
    ...     prop_id="pinx",
    ...     trans_id="sale_id",
    ...     price="sale_price",
    ...     min_period_dist=12,  # minimum months between sales
    ... )

Creating the Index
------------------

Once your data is prepared, you can create the repeat sales index:

.. code-block:: python

    >>> from hpipy.price_index import RepeatTransactionIndex

    # Create the index.
    >>> hpi = RepeatTransactionIndex.create_index(
    ...     trans_data=trans_data,
    ...     prop_id="pinx",
    ...     trans_id="sale_id",
    ...     price="sale_price",
    ...     date="sale_date",
    ...     estimator="robust",  # or "base", "weighted"
    ...     log_dep=True,  # use log of price differences
    ...     smooth=True,  # apply smoothing to the index
    ... )

Parameters
----------

The main parameters for repeat sales index creation are:

.. admonition:: Parameters
   :class: hint

   **estimator** : str
       The type of estimator to use:

       * "base": Standard OLS estimation
       * "robust": Robust regression (less sensitive to outliers)
       * "weighted": Weighted regression based on time between sales

   **log_dep** : bool
       Whether to use log price differences (recommended).

Advanced Usage
--------------

For more control over the index creation process, you can use the lower-level API:

.. code-block:: python

    >>> from hpipy.price_index import RepeatTransactionIndex
    >>> from hpipy.price_model import RepeatTransactionModel

    # Create and fit the model.
    >>> model = RepeatTransactionModel(
    ...     trans_data,
    ...     estimator="robust",
    ...     log_dep=True,
    ... ).fit()

    # Create the index.
    >>> hpi = RepeatTransactionIndex.from_model(
    ...     model, trans_data=trans_data, smooth=True
    ... )

Evaluating the Index
--------------------

You can evaluate the index quality using various metrics:

.. code-block:: python

    >>> from hpipy.utils.metrics import accuracy, volatility
    >>> from hpipy.utils.plotting import plot_index

    # Calculate accuracy.
    >>> acc = accuracy(hpi)

    # Calculate volatility.
    >>> vol = volatility(hpi)

    # Visualize the index.
    >>> plot_index(hpi, smooth=True).properties(title="Repeat Sales Index")
    alt.LayerChart(...)

.. invisible-altair-plot::

    from hpipy.datasets import load_ex_sales
    from hpipy.period_table import PeriodTable
    from hpipy.trans_data import RepeatTransactionData
    from hpipy.price_index import RepeatTransactionIndex
    from hpipy.price_model import RepeatTransactionModel
    from hpipy.utils.plotting import plot_index

    df = load_ex_sales()
    sales_hdata = PeriodTable(df).create_period_table("sale_date", periodicity="monthly")
    trans_data = RepeatTransactionData(sales_hdata).create_transactions(
        prop_id="pinx", trans_id="sale_id", price="sale_price", min_period_dist=12
    )
    model = RepeatTransactionModel(trans_data, estimator="robust", log_dep=True).fit()
    hpi = RepeatTransactionIndex.from_model(model, trans_data=trans_data, smooth=True)
    chart = plot_index(hpi, smooth=True).properties(title="Repeat Sales Index", width=600)
