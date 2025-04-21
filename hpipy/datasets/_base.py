import importlib.resources as pkg_resources

import pandas as pd


def load_ex_sales() -> pd.DataFrame:
    """Load example home sales dataset.

    Returns:
        pandas.DataFrame: A DataFrame with house sales.

    Example:
        >>> from hpipy.datasets import load_ex_sales
        >>> df = load_ex_sales()
        >>> df.head()
                pinx      sale_id  sale_price  ... eff_age   longitude   latitude
        0  ..0007600046   2011..2621      308900  ...      12 -122.302032  47.603913
        1  ..0007600054  2010..16414      369950  ...     103 -122.302030  47.603044
        2  ..0007600057  2014..23738      520000  ...     112 -122.302114  47.602875
        3  ..0007600057  2016..28612      625000  ...     114 -122.302114  47.602875
        4  ..0007600065  2014..15956      465000  ...       0 -122.297278  47.601812

    """
    with pkg_resources.files("hpipy.datasets.data").joinpath("ex_sales.csv").open("r") as f:
        return pd.read_csv(f, parse_dates=["sale_date"])


def load_seattle_sales() -> pd.DataFrame:
    """Load Seattle home sales dataset.

    Returns:
        pandas.DataFrame: A DataFrame with house sales.

    Example:
        >>> from hpipy.datasets import load_seattle_sales
        >>> df = load_seattle_sales()
        >>> df.head()
                pinx      sale_id  sale_price  ... eff_age   longitude   latitude
        0  ..0001800010   2013..2432      289000  ...       6 -122.312491  47.561380
        1  ..0001800066  2013..21560      356000  ...      87 -122.322007  47.550353
        2  ..0001800075  2010..24221      333500  ...      80 -122.311654  47.561470
        3  ..0001800075   2016..6629      577200  ...      86 -122.311654  47.561470
        4  ..0001800080   2012..9521      237000  ...      72 -122.309695  47.561472

    """
    with pkg_resources.files("hpipy.datasets.data").joinpath("seattle_sales.csv").open("r") as f:
        return pd.read_csv(f, parse_dates=["sale_date"])
