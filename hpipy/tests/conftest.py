"""Configure unit tests."""

import pandas as pd
import pytest

from hpipy.datasets import load_ex_sales, load_seattle_sales


class MyTestDataset:
    """Test dataset objects."""

    def __init__(self) -> None:
        """Initialize toy dataset class."""

    @property
    def dataset(self) -> pd.DataFrame:
        """Dataset."""
        return load_ex_sales()


@pytest.fixture(scope="session")
def toy_dataset() -> pd.DataFrame:
    """Return toy dataset class instance."""
    return MyTestDataset().dataset


class SeattleDataset:
    """Test dataset objects."""

    def __init__(self) -> None:
        """Initialize Seattle dataset class."""

    @property
    def dataset(self) -> pd.DataFrame:
        """Dataset."""
        return load_seattle_sales()


@pytest.fixture(scope="session")
def seattle_dataset() -> pd.DataFrame:
    """Return Seattle dataset class instance."""
    return SeattleDataset().dataset
