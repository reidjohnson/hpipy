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

.. toctree::
   :maxdepth: 1
   :caption: Quick Start
   :hidden:

   Basic Usage <self>
   data_format
   available_methods
   next_steps
