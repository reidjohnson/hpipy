import numpy as np
import pandas as pd
import pytest
from altair import Chart, HConcatChart, LayerChart, VConcatChart

from hpipy.price_index import BaseHousePriceIndex
from hpipy.utils.plotting import (
    plot_index,
    plot_index_accuracy,
    plot_index_volatility,
    plot_series_revision,
    plot_series_volatility,
)


class MockHPI(BaseHousePriceIndex):
    """Mock HPI class for testing."""

    def __init__(self, periods=None, values=None, imputed=None) -> None:
        """Initialize mock HPI."""
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        self.periods = pd.Series(range(12), name="period") if periods is None else periods
        self.value = pd.Series(np.random.randn(12), name="value") if values is None else values
        self.imputed = pd.Series(np.zeros(12), name="imputed") if imputed is None else imputed
        self.smooth = pd.Series(np.random.randn(12), name="smooth")
        self.model = type(
            "MockModel",
            (),
            {
                "period_table": pd.DataFrame(
                    {
                        "period": range(12),
                        "name": [f"2023-{m}" for m in months],
                    },
                ),
            },
        )

    def _create_model(self) -> None:
        """Mock create model method."""

    def _create_transactions(self):
        """Mock create transactions method."""
        return pd.DataFrame(
            {
                "price": np.random.randn(100),
                "period": np.random.randint(0, 12, 100),
            },
        )


class MockHPISeries:
    """Mock HPI series class for testing."""

    def __init__(self, n_series=3) -> None:
        """Initialize mock HPI series."""
        self.hpis = [MockHPI() for _ in range(n_series)]

        # Create revision data.
        periods = range(12)
        n_revisions = 3
        self.revision = pd.DataFrame(
            np.random.randn(len(periods), n_revisions),
            index=periods,
            columns=[f"rev_{i}" for i in range(n_revisions)],
        )
        self.revision_smooth = pd.DataFrame(
            np.random.randn(len(periods), n_revisions),
            index=periods,
            columns=[f"rev_{i}" for i in range(n_revisions)],
        )
        self.revision_mean = pd.Series(np.random.randn(len(periods)), index=periods, name="mean")
        self.revision_median = pd.Series(
            np.random.randn(len(periods)),
            index=periods,
            name="median",
        )


@pytest.fixture
def mock_hpi():
    """Create a mock HPI instance."""
    return MockHPI()


@pytest.fixture
def mock_hpi_series():
    """Create a mock HPI series instance."""
    return MockHPISeries()


@pytest.fixture
def error_df():
    """Create a mock error DataFrame."""
    return pd.DataFrame(
        {
            "pred_period": np.repeat(range(12), 10),
            "error": np.random.randn(120),
        },
    )


@pytest.fixture
def volatility_df():
    """Create a mock volatility DataFrame."""
    index = pd.Series(range(12))
    return pd.DataFrame(
        {
            "roll": np.random.randn(12),
            "mean": np.random.randn(12),
            "median": np.random.randn(12),
        },
        index=index,
    )


def test_plot_index_basic(mock_hpi) -> None:
    """Test basic plot_index functionality."""
    chart = plot_index(mock_hpi)
    assert isinstance(chart, Chart)
    assert chart.to_dict()["height"] == 300
    assert chart.to_dict()["width"] == 800


def test_plot_index_with_imputed(mock_hpi) -> None:
    """Test plot_index with imputed values shown."""
    mock_hpi.imputed = pd.Series(np.ones(12), name="imputed")
    chart = plot_index(mock_hpi, show_imputed=True)
    assert isinstance(chart, LayerChart)
    assert len(chart.layer) == 2  # base line and imputed points


def test_plot_index_with_smooth(mock_hpi) -> None:
    """Test plot_index with smoothing."""
    chart = plot_index(mock_hpi, smooth=True)
    assert isinstance(chart, LayerChart)
    assert len(chart.layer) == 2  # base line and smooth line


def test_plot_index_accuracy_basic(error_df) -> None:
    """Test basic plot_index_accuracy functionality."""
    chart = plot_index_accuracy(error_df)
    # Check that chart is a combination of multiple charts.
    assert isinstance(chart, VConcatChart)

    # Check the subcharts.
    assert isinstance(chart.vconcat[0], HConcatChart)
    assert isinstance(chart.vconcat[1], HConcatChart)

    # Check dimensions of first subchart.
    assert chart.vconcat[0].hconcat[0].to_dict()["height"] == 150
    assert chart.vconcat[0].hconcat[0].to_dict()["width"] == 375


def test_plot_index_accuracy_custom_size(error_df) -> None:
    """Test plot_index_accuracy with custom size."""
    chart = plot_index_accuracy(error_df, size=5)
    assert isinstance(chart, VConcatChart)

    # Check the subcharts.
    assert isinstance(chart.vconcat[0], HConcatChart)
    assert isinstance(chart.vconcat[1], HConcatChart)


def test_plot_index_volatility_basic(volatility_df) -> None:
    """Test basic plot_index_volatility functionality."""
    chart = plot_index_volatility(volatility_df)
    assert isinstance(chart, LayerChart)

    # Check dimensions of base chart.
    assert chart.to_dict()["height"] == 300
    assert chart.to_dict()["width"] == 800


def test_plot_series_volatility_basic(mock_hpi_series) -> None:
    """Test basic plot_series_volatility functionality."""
    chart = plot_series_volatility(mock_hpi_series)
    assert isinstance(chart, LayerChart)
    assert chart.to_dict()["height"] == 300
    assert chart.to_dict()["width"] == 800


def test_plot_series_volatility_with_smooth(mock_hpi_series) -> None:
    """Test plot_series_volatility with smoothing."""
    chart = plot_series_volatility(mock_hpi_series, smooth=True)
    assert isinstance(chart, LayerChart)


def test_plot_series_revision_basic(mock_hpi_series) -> None:
    """Test basic plot_series_revision functionality."""
    chart = plot_series_revision(mock_hpi_series)
    assert isinstance(chart, LayerChart)


def test_plot_series_revision_with_smooth(mock_hpi_series) -> None:
    """Test plot_series_revision with smoothing."""
    chart = plot_series_revision(mock_hpi_series, smooth=True)
    assert isinstance(chart, LayerChart)


def test_plot_series_revision_custom_measure(mock_hpi_series) -> None:
    """Test plot_series_revision with custom measure."""
    chart = plot_series_revision(mock_hpi_series, measure="mean")
    assert isinstance(chart, LayerChart)
