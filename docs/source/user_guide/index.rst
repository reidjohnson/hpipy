.. _user_guide/index:

User Guide
==========

Welcome to the **hpiPy User Guide**. This guide walks you through creating and evaluating **House Price Indices** using various methods—from repeat sales to hedonic pricing to machine learning.

Index Construction Methods
--------------------------

Use one of the following methods to build house price indices:

.. grid:: 2
   :gutter: 2
   :margin: 2

   .. grid-item-card:: 📈 Repeat Sales
      :link: repeat_sales
      :link-type: doc

      Build house price indices by pairing repeat sales of unchanged properties.

   .. grid-item-card:: 🏘️ Hedonic Pricing
      :link: hedonic_pricing
      :link-type: doc

      Model house price indices as a function of house features like size, location, and age.

   .. grid-item-card:: 🌲 Random Forest
      :link: random_forest
      :link-type: doc

      Use an ensemble of decision trees to learn complex, non-linear price patterns.

   .. grid-item-card:: 🧠 Neural Network
      :link: neural_network
      :link-type: doc

      Apply a deep learning model to learn complex, non-linear price patterns.

Evaluation & Comparison
-----------------------

Once you've created your indices, use the following tools to evaluate and compare methods:

.. grid:: 2
   :gutter: 2
   :margin: 2

   .. grid-item-card:: Accuracy Metrics
      :link: accuracy_metrics
      :link-type: doc

      Measure how well an index predicts actual property values.

   .. grid-item-card:: Volatility Metrics
      :link: volatility_metrics
      :link-type: doc

      Quantify the smoothness and stability of an index over time.

   .. grid-item-card:: Revision Metrics
      :link: revision_metrics
      :link-type: doc

      Track how index values change with new data over time.

   .. grid-item-card:: Comparative Analysis
      :link: comparative_analysis
      :link-type: doc

      Compare multiple index construction methods side by side.

   .. grid-item-card:: Time Window Analysis
      :link: time_window_analysis
      :link-type: doc

      Evaluate metrics across different rolling time windows.

.. toctree::
   :hidden:
   :caption: Price Index Methods
   :maxdepth: 1

   repeat_sales
   hedonic_pricing
   random_forest
   neural_network

.. toctree::
   :hidden:
   :caption: Evaluation & Comparison
   :maxdepth: 1

   accuracy_metrics
   volatility_metrics
   revision_metrics
   comparative_analysis
   time_window_analysis
