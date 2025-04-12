Evaluation Metrics
==================

hpiPy provides a comprehensive set of metrics to evaluate house price indices, focusing on accuracy, volatility, and revision metrics. These metrics help compare different methods and assess index quality.

Accuracy Metrics
----------------

Accuracy metrics measure how well the index predicts actual property values:

.. code-block:: python

    from hpipy.utils.metrics import accuracy
    from hpipy.utils.plotting import plot_index_accuracy

    # Calculate accuracy for a single index.
    acc = accuracy(index)

    # Plot accuracy metrics.
    plot_index_accuracy(index)

Volatility Metrics
------------------

Volatility metrics measure the stability and smoothness of the index:

.. code-block:: python

    from hpipy.utils.metrics import volatility
    from hpipy.utils.plotting import plot_index_volatility

    # Calculate volatility for a single index.
    vol = volatility(index)

    # Plot volatility metrics.
    plot_index_volatility(index)

Revision Metrics
----------------

Revision metrics assess how index values change as new data becomes available:

.. code-block:: python

    from hpipy.utils.metrics import revision

    # Calculate revision metrics.
    rev = revision(index_series)

Comparative Analysis
--------------------

Compare different index methods:

.. code-block:: python

    # Create indices using different methods.
    rt_index = RepeatTransactionIndex.create_index(...)
    hed_index = HedonicIndex.create_index(...)
    rf_index = RandomForestIndex.create_index(...)
    nn_index = NeuralNetworkIndex.create_index(...)

    # Compare accuracy.
    accuracies = [accuracy(index) for index in [rt_index, hed_index, rf_index, nn_inde]]

    # Compare volatility.
    volatilities = [volatility(index) for index in [rt_index, hed_index, rf_index, nn_inde]]

Time Window Analysis
--------------------

Analyze metrics over different time windows:

.. code-block:: python

    # Calculate rolling metrics.
    rolling_acc = accuracy(index, window=12)  # 12-month window
    rolling_vol = volatility(index, window=12)
