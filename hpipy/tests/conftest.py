"""Configure unit tests."""

import os

import pandas as pd
import pytest

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

TEST_DATA_DIR = os.path.join(BASE_PATH, "../../data")
TOY_DATA_FILE = "ex_sales.csv"
SEATTLE_DATA_FILE = "seattle_sales.csv"


class MyTestDataset:
    """Test dataset objects."""

    def __init__(self) -> None:
        """Initialize toy dataset class."""
        self._dataset = os.path.join(TEST_DATA_DIR, TOY_DATA_FILE)

    @property
    def dataset(self) -> pd.DataFrame:
        """Dataset."""
        return pd.read_csv(self._dataset, parse_dates=["sale_date"])


@pytest.fixture(scope="session")
def toy_dataset() -> pd.DataFrame:
    """Return toy dataset class instance."""
    return MyTestDataset().dataset


class SeattleDataset:
    """Test dataset objects."""

    def __init__(self) -> None:
        """Initialize Seattle dataset class."""
        self._dataset = os.path.join(TEST_DATA_DIR, SEATTLE_DATA_FILE)

    @property
    def dataset(self) -> pd.DataFrame:
        """Dataset."""
        return pd.read_csv(self._dataset, parse_dates=["sale_date"])


@pytest.fixture(scope="session")
def seattle_dataset() -> pd.DataFrame:
    """Return Seattle dataset class instance."""
    return SeattleDataset().dataset
