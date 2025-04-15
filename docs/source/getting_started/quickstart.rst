.. _quickstart:

Quick Start
===========

This guide will help you get started with hpiPy quickly.

Basic Usage
-----------

Here's a simple example of creating a house price index using repeat sales data:

.. code-block:: python

    >>> import pandas as pd
    >>> from hpipy.price_index import RepeatTransactionIndex
    >>> from hpipy.utils.plotting import plot_index

    # Load your sales data.
    >>> df = pd.read_csv("data/seattle_sales.csv", parse_dates=["sale_date"])

    # Create an index.
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

    # Visualize the index.
    >>> plot_index(hpi, smooth=True).properties(width=600, title="Example Index")
    alt.LayerChart(...)

.. invisible-altair-plot::

    import pandas as pd
    from hpipy.price_index import RepeatTransactionIndex
    from hpipy.utils.plotting import plot_index
    df = pd.read_csv("data/seattle_sales.csv", parse_dates=["sale_date"])
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
    chart = plot_index(hpi, smooth=True).properties(width=600, title="Example Index")


Data Format
-----------

Your input data should be a pandas DataFrame with the following columns:

* A date column (e.g., "sale_date")
* A price column (e.g., "sale_price")
* A property identifier column (e.g., "pinx")
* A transaction identifier column (e.g., "sale_id")

Example data structure:

.. code-block:: python

    >>> import pandas as pd
    >>> df = pd.read_csv("data/ex_sales.csv", parse_dates=["sale_date"])
    >>> df.iloc[:, :4].head()
               pinx      sale_id  sale_price  sale_date
    0  ..0007600046   2011..2621      308900 2011-02-22
    1  ..0007600054  2010..16414      369950 2010-08-24
    2  ..0007600057  2014..23738      520000 2014-08-05
    3  ..0007600057  2016..28612      625000 2016-08-22
    4  ..0007600065  2014..15956      465000 2014-06-05

Available Methods
-----------------

hpiPy supports several methods for creating house price indices:

1. Repeat Transaction Method
   
   .. code-block:: python

      from hpipy.price_index import RepeatTransactionIndex
      hpi = RepeatTransactionIndex.create_index(...)

2. Hedonic Price Method
   
   .. code-block:: python

      from hpipy.price_index import HedonicPriceIndex
      hpi = HedonicPriceIndex.create_index(...)

3. Random Forest Method
   
   .. code-block:: python

      from hpipy.extensions import RandomForestIndex
      hpi = RandomForestIndex.create_index(...)

4. Neural Network Method
   
   .. code-block:: python

      from hpipy.extensions import NeuralNetworkIndex
      hpi = NeuralNetworkIndex.create_index(...)

Next Steps
----------

* Check out the :doc:`../user_guide/index` for detailed information.
* See the :doc:`../api/index` reference for complete documentation.
