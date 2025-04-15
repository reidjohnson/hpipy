.. data_format:

Data Format
===========

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
