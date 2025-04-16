.. available_methods:

Available Methods
=================

hpiPy supports several methods for creating house price indices:

1. :doc:`../user_guide/repeat_sales`

   .. code-block:: python

      from hpipy.price_index import RepeatTransactionIndex
      hpi = RepeatTransactionIndex.create_index(...)

2. :doc:`../user_guide/hedonic_pricing`

   .. code-block:: python

      from hpipy.price_index import HedonicPriceIndex
      hpi = HedonicPriceIndex.create_index(...)

3. :doc:`../user_guide/random_forest`

   .. code-block:: python

      from hpipy.extensions import RandomForestIndex
      hpi = RandomForestIndex.create_index(...)

4. :doc:`../user_guide/neural_network`

   .. code-block:: python

      from hpipy.extensions import NeuralNetworkIndex
      hpi = NeuralNetworkIndex.create_index(...)
