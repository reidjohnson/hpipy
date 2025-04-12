.. _quickstart:

Quickstart
==========

This guide will help you get started with hpiPy quickly.

Basic Usage
-----------

Here's a simple example of creating a house price index using repeat sales data:

.. code-block:: python

    import pandas as pd
    from hpipy.price_index import RepeatTransactionIndex

    # Load your sales data.
    df = pd.read_csv("sales_data.csv", parse_dates=["sale_date"])

    # Create an index.
    hpi = RepeatTransactionIndex.create_index(
        trans_data=df,
        date="sale_date",
        price="sale_price",
        prop_id="pinx",
        trans_id="sale_id",
        estimator="robust",
        log_dep=True,
        smooth=True,
    )

    # Print the index.
    print(hpi)

Data Format
-----------

Your input data should be a pandas DataFrame with the following columns:

* A date column (e.g., "sale_date")
* A price column (e.g., "sale_price")
* A property identifier column (e.g., "pinx")
* A transaction identifier column (e.g., "sale_id")

Example data structure:

.. code-block:: python

    >>> df.head()
       sale_id    sale_date  sale_price    pinx
    0        1  2010-01-15     250000  prop_01
    1        2  2010-02-01     300000  prop_02
    2        3  2012-06-15     275000  prop_01
    3        4  2012-07-01     320000  prop_02
    4        5  2010-03-15     280000  prop_03

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

3. Neural Network Method
   
   .. code-block:: python

      from hpipy.extensions import NeuralNetworkIndex
      hpi = NeuralNetworkIndex.create_index(...)

Next Steps
----------

* Check out the :doc:`user_guide/index` for detailed information.
* See the :doc:`api/index` reference for complete documentation.
