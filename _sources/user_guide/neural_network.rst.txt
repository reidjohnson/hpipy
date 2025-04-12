Neural Network Method
=====================

The Neural Network method in hpiPy provides a deep learning approach to house price index creation, capable of learning complex patterns and separating property-specific effects from market-level trends.

Overview
--------

The Neural Network approach:

1. Separates property-specific and market-level effects
2. Learns non-linear relationships automatically
3. Can handle high-dimensional feature spaces
4. Supports both local and global market patterns

Data Preparation
----------------

Required data structure:

* A date column (e.g., "sale_date")
* A price column (e.g., "sale_price")
* Property characteristics
* A transaction identifier

Example setup:

.. code-block:: python

    import pandas as pd
    from hpipy.period_table import PeriodTable
    from hpipy.extensions import NeuralNetworkIndex

    # Load and prepare data.
    df = pd.read_csv("sales_data.csv", parse_dates=["sale_date"])
    
    # Create period table.
    trans_data = PeriodTable(df).create_period_table(
        "sale_date",
        periodicity="monthly",
    )

Creating the Index
------------------

Create a Neural Network-based index:

.. code-block:: python

    # Create the index.
    hpi = NeuralNetworkIndex.create_index(
        trans_data=trans_data,
        trans_id="sale_id",
        prop_id="pinx",
        date="sale_date",
        price="sale_price",
        dep_var="price",
        ind_var=["tot_sf", "beds", "baths"],
        estimator="attributional",
        feature_dict={
            "numerics": [],
            "log_numerics": ["tot_sf"],
            "categoricals": [],
            "ordinals": ["beds", "baths"],
            "hpi": ["sale_date"],
        },
    )

Network Architecture
--------------------

The neural network consists of:

1. Property Characteristics Network:
   * Processes property features
   * Learns property-specific value components

2. Time Effect Network:
   * Captures temporal market trends
   * Generates the house price index

3. Combined Output:
   * Merges property and time effects
   * Produces final price predictions

Parameters
----------

Key neural network parameters:

estimator : str
    Estimator type. "residual" or "attributional". Defaults to "residual".

feature_dict : dict
    Feature dictionary.

hidden_layers : list
    List of integers specifying the number of neurons in each hidden layer.

activation : str
    Activation function to use ("relu", "tanh", etc.).

learning_rate : float
    Learning rate for optimization.

epochs : int
    Number of training epochs.

batch_size : int
    Batch size for training.

smooth : bool
    Whether to apply smoothing to the final index.

dropout : float
    Dropout rate for regularization (0 to 1).

Evaluating the Index
--------------------

Evaluate the random forest index using various metrics:

.. code-block:: python

    from hpipy.utils.metrics import volatility
    from hpipy.utils.plotting import plot_index

    # Calculate metrics.
    vol = volatility(hpi)

    # Visualize results.
    plot_index(hpi)
