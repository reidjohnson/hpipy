Neural Network
==============

The Neural Network method in hpiPy provides a deep learning approach to house price index creation, capable of learning complex patterns and separating property-specific effects from market-level trends.

Overview
--------

The Neural Network method:

1. Separates property-specific and market-level effects
2. Learns non-linear relationships automatically
3. Can handle high-dimensional feature spaces
4. Supports both local and global market patterns

The Neural Network supports two distinct approaches for extracting the house price index.

Residual Approach
~~~~~~~~~~~~~~~~~

The residual approach extracts the index directly from the market pathway of the neural network. It works by:

1. Training the network on the full feature set
2. During index extraction, zeroing out all non-temporal features
3. Examining the network's output to capture the "residual" market trend
4. Computing the index from these isolated temporal effects

This approach is:

* More computationally efficient
* Direct in its interpretation
* The default approach in hpiPy

Attributional Approach
~~~~~~~~~~~~~~~~~~~~~~

The attributional approach derives the index through explainability analysis of both market and time components. It:

1. Trains the network on the full feature set
2. Uses attribution techniques to decompose predictions
3. Quantifies how much price change is due to temporal vs. market factors
4. Computes the index from the attributed temporal components

This approach offers:

* Higher computational complexity
* More granular decomposition of effects

Data Preparation
----------------

Required data structure:

* A date column (e.g., "sale_date")
* A price column (e.g., "sale_price")
* Property characteristics
* A transaction identifier

Example setup:

.. code-block:: python

    >>> import pandas as pd
    >>> from hpipy.datasets import load_ex_sales
    >>> from hpipy.period_table import PeriodTable
    >>> from hpipy.trans_data import HedonicTransactionData

    # Load and prepare data.
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

Create a Neural Network-based index using either approach:

.. code-block:: python

    >>> from hpipy.extensions import NeuralNetworkIndex

    >>> kwargs = {
    ...     "prop_id": "pinx",
    ...     "trans_id": "sale_id",
    ...     "price": "sale_price",
    ...     "date": "sale_date",
    ...     "dep_var": "price",
    ...     "ind_var": ["tot_sf", "beds", "baths"],
    ...     "feature_dict": {
    ...         "numerics": [],
    ...         "log_numerics": ["tot_sf"],
    ...         "categoricals": [],
    ...         "ordinals": ["beds", "baths"],
    ...         "hpi": ["sale_date"],
    ...     },
    ...     "preprocess_geo": False,
    ...     "random_seed": 0,
    ... }

    # Create index using residual approach (default)
    >>> hpi_residual = NeuralNetworkIndex.create_index(
    ...     trans_data=trans_data,
    ...     estimator="residual",  # default
    ...     **kwargs,
    ... )

    # Create index using attributional approach
    >>> hpi_attributional = NeuralNetworkIndex.create_index(
    ...     trans_data=trans_data,
    ...     estimator="attributional",
    ...     **kwargs,
    ... )

Network Architecture
--------------------

The neural network consists of:

1. Property Characteristics Network

  * Processes property features
  * Learns property-specific value components
  * Handles both continuous and categorical inputs
  * Uses embeddings for categorical features

2. Time Effect Network

  * Captures temporal market trends
  * Generates the house price index
  * Processes temporal features
  * Learns market-level patterns

3. Combined Output

  * Merges property and time effects
  * Produces final price predictions
  * Balances feature importance
  * Enables index extraction

Parameters
----------

Key neural network parameters:

dep_var : str
    Dependent variable to model.

ind_var : list
    Independent variables to use in the model.

estimator : str
    Estimator type. Choose between:

    * "residual": Extracts index from market pathway (default)
    * "attributional": Derives index through explainability analysis

feature_dict : dict
    Feature dictionary specifying how different variables should be processed:

    * numerics: Standard numeric features
    * log_numerics: Features to be log-transformed
    * categoricals: Categorical features for embedding
    * ordinals: Ordinal features
    * hpi: Temporal features for index generation

num_models : int
    Number of models to train in ensemble.

num_epochs : int
    Number of training epochs.

batch_size : int
    Batch size for training.

hidden_dims : list
    List of integers specifying the number of neurons in each hidden layer.

emb_size : int
    Embedding size for categorical features.

dropout_rate : float
    Dropout rate for regularization (0 to 1).

learning_rate : float
    Learning rate for optimization.

Evaluating the Index
--------------------

Evaluate the neural network index using various metrics:

.. code-block:: python

    >>> import altair as alt
    >>> from hpipy.utils.metrics import volatility
    >>> from hpipy.utils.plotting import plot_index

    # Calculate metrics.
    >>> vol_residual = volatility(hpi_residual)
    >>> vol_attributional = volatility(hpi_attributional)

    # Visualize results.
    >>> alt.layer(
    ...     (
    ...         plot_index(hpi_residual)
    ...         .transform_calculate(method="'Residual'")
    ...         .encode(color=alt.Color("method:N", title="Method"))
    ...     ),
    ...     (
    ...         plot_index(hpi_attributional)
    ...         .transform_calculate(method="'Attributional'")
    ...         .encode(color=alt.Color("method:N", title="Method"))
    ...     ),
    ... ).properties(title="Neural Network Index")
    alt.LayerChart(...)

.. invisible-altair-plot::

    import altair as alt
    import pandas as pd
    from hpipy.datasets import load_ex_sales
    from hpipy.extensions import NeuralNetworkIndex
    from hpipy.period_table import PeriodTable
    from hpipy.trans_data import HedonicTransactionData
    from hpipy.utils.plotting import plot_index
    df = load_ex_sales()
    sales_hdata = PeriodTable(df).create_period_table("sale_date", periodicity="monthly")
    trans_data = HedonicTransactionData(sales_hdata).create_transactions(
        prop_id="pinx", trans_id="sale_id", price="sale_price"
    )
    kwargs = {
        "prop_id": "pinx",
        "trans_id": "sale_id",
        "price": "sale_price",
        "date": "sale_date",
        "dep_var": "price",
        "ind_var": ["tot_sf", "beds", "baths"],
        "feature_dict": {
            "numerics": [],
            "log_numerics": ["tot_sf"],
            "categoricals": [],
            "ordinals": ["beds", "baths"],
            "hpi": ["sale_date"],
        },
        "preprocess_geo": False,
        "random_seed": 0,
    }
    hpi_residual = NeuralNetworkIndex.create_index(
        trans_data=trans_data, estimator="residual", **kwargs
    )
    hpi_attributional = NeuralNetworkIndex.create_index(
        trans_data=trans_data, estimator="attributional", **kwargs
    )
    chart = alt.layer(
        (
            plot_index(hpi_residual)
            .transform_calculate(method="'Residual'")
            .encode(color=alt.Color("method:N", title="Method"))
        ),
        (
            plot_index(hpi_attributional)
            .transform_calculate(method="'Attributional'")
            .encode(color=alt.Color("method:N", title="Method"))
        ),
    ).properties(title="Neural Network Index", width=525)
