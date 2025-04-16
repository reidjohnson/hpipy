import importlib.resources as pkg_resources

import pandas as pd


def load_ex_sales() -> pd.DataFrame:
    """Load example home sales dataset.

    Returns:
        pandas.DataFrame: A DataFrame with house sales.
    """
    with pkg_resources.files("hpipy.datasets.data").joinpath("ex_sales.csv").open("r") as f:
        return pd.read_csv(f, parse_dates=["sale_date"])


def load_seattle_sales() -> pd.DataFrame:
    """Load Seattle home sales dataset.

    Returns:
        pandas.DataFrame: A DataFrame with house sales.
    """
    with pkg_resources.files("hpipy.datasets.data").joinpath("seattle_sales.csv").open("r") as f:
        return pd.read_csv(f, parse_dates=["sale_date"])
