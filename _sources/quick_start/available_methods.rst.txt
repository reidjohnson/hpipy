.. available_methods:

Available Methods
=================

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
