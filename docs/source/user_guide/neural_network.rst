Bifurcated Neural Network
=========================

The neural network method offers a deep learning–based approach to constructing house price indices. It automatically learns complex, nonlinear relationships between property features and sale prices, enabling it to separate individual property effects from broader market trends. While it requires large volumes of sales and property data, it is particularly powerful at scale and can produce accurate, fine-grained, and timely indices without relying on repeat sales.

.. note::

    Background on model construction for the neural network method can be found in:

    Krause and Johnson (2024), "A Multi-Criteria Evaluation of House Price Indexes". `https://github.com/andykrause/hpi_research/tree/master/papers/hpi_comp <https://github.com/andykrause/hpi_research/tree/master/papers/hpi_comp>`_.

Overview
--------

The neural network method:

1. Separates property-specific and market-level effects
2. Learns non-linear relationships automatically
3. Can handle high-dimensional feature spaces
4. Supports both local and global market patterns

The neural network method supports two distinct approaches for extracting the house price index:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - Residual Approach
     - Attributional Approach
   * - Description
     - Extracts the index directly from the market pathway by isolating temporal market effects.
     - Uses explainability techniques (e.g., SHAP )to attribute prediction components to market-level factors.
   * - Steps
     - 1. Train on full feature set
       2. Zero out non-market (i.e., property-specific) feautures
       3. Extract residual market effects
       4. Convert to index
     - 1. Train on full feature set
       2. Apply explainability techniques to assign feature attributions
       3. Extract market-level attributions
       4. Convert to index
   * - Benefits
     - * Explicit separation of effects
       * Structural interpretation
       * Default in `hpiPy`
     - * Granular effect decomposition
       * Feature-level interpretability
   * - Drawbacks
     - * Less flexible than attributional
     - * Relies on post-hoc attributions

**Attributional Approach**

The attributional approach decomposes the logarithm of a property's value into a market-level price index and a property-specific component. This reflects the idea that housing value is jointly determined by macroeconomic conditions and the characteristics of the property itself (`ref <ref-lusk_>`_):

.. math::

    \log V_{it} = \log P_t + \log Q_i + \varepsilon_{it}

where:

- :math:`V_{it}` is the observed transaction price (or value) of property *i* at time *t*.
- :math:`P_t` is the market-level price index at time *t*, common to all properties.
- :math:`Q_i` is the time-invariant quality or quantity of property *i* (e.g., structural/locational attributes).
- :math:`\varepsilon_{it}` is a residual term capturing idiosyncratic noise or omitted effects.

This model is conceptually similar to hedonic or repeat-sales approaches, where market effects and property characteristics are disentangled.

**Residual Approach**

The residual approach models the house price as a black-box prediction that integrates market and property factors, and then uses explainability methods to decompose this prediction into attributions. Specifically, DeepLIFT attributes the model output to individual features relative to a reference (baseline) input (`ref <ref-deeplift_>`_):

.. math::

    \hat{V}_i = f(x_i)

.. math::

    \Delta \hat{V}_i = \hat{V}_i - \hat{V}_i^{\text{ref}} = \sum_{j} C_j

.. math::

    C_j = m_j \cdot \Delta x_{ij}

where:

- :math:`\hat{V}_i = f(x_i)` is the model’s predicted value for property *i*.
- :math:`x_i` is the feature vector describing property *i* (e.g., square footage, year built, etc.).
- :math:`\hat{V}_i^{\text{ref}} = f(x_i^{\text{ref}})` is the prediction for a baseline (e.g., average, median, or zeroed) property.
- :math:`\Delta \hat{V}_i` is the total difference in predicted value from the baseline.
- :math:`C_j` is the contribution of feature :math:`j`, computed as the product of the feature’s difference from baseline, :math:`\Delta x_{ij} = x_{ij} - x_{j}^{\text{ref}}`, and its multiplier :math:`m_j`, which represents the sensitivity of the output to that feature.

This approach allows for interpretability of complex nonlinear models by expressing the prediction in terms of feature-level contributions.

.. _ref-lusk: https://lusk.usc.edu/research/working-papers/revisiting-past-revision-repeat-sales-and-hedonic-indexes-house-prices
.. _ref-deeplift: https://arxiv.org/abs/1704.02685

Data Preparation
----------------

Required data structure:

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

Create a neural network-based index using either approach:

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

    # Create index using residual approach (default).
    >>> hpi_residual = NeuralNetworkIndex.create_index(
    ...     trans_data=trans_data,
    ...     estimator="residual",  # default
    ...     **kwargs,
    ... )

    # Create index using attributional approach.
    >>> hpi_attributional = NeuralNetworkIndex.create_index(
    ...     trans_data=trans_data,
    ...     estimator="attributional",
    ...     **kwargs,
    ... )

Parameters
----------

The main parameters for neural network index creation are:

.. admonition:: Parameters
   :class: hint

   **dep_var** : str
       Dependent variable to model.

   **ind_var** : list
       Independent variables to use in the model.

   **estimator** : str
       Estimator type. Choose between:

       * "residual": Extracts index from market pathway (default)
       * "attributional": Derives index through explainability analysis

   **feature_dict** : dict
       Feature dictionary specifying how different variables should be processed:

       * numerics: Standard numeric features
       * log_numerics: Features to be log-transformed
       * categoricals: Categorical features for embedding
       * ordinals: Ordinal features
       * hpi: Temporal features for index generation

   **num_models** : int
       Number of models to train in ensemble.

   **num_epochs** : int
       Number of training epochs.

   **batch_size** : int
       Batch size for training.

   **hidden_dims** : list
       List of integers specifying the number of neurons in each hidden layer.

   **emb_size** : int
       Embedding size for categorical features.

   **dropout_rate** : float
       Dropout rate for regularization (0 to 1).

   **learning_rate** : float
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

    # Visualize the index.
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
