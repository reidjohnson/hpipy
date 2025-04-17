import copy

import numpy as np
import pandas as pd
import pytest

from hpipy.price_index import BaseHousePriceIndex, HedonicIndex, RepeatTransactionIndex
from hpipy.price_model import HedonicModel, RepeatTransactionModel
from hpipy.trans_data import HedonicTransactionData, RepeatTransactionData


@pytest.mark.usefixtures("seattle_dataset")
def test_rt_create_trans_toy(seattle_dataset: pd.DataFrame) -> None:
    """Test repeat transaction index creation with toy dataset."""
    full_1 = RepeatTransactionIndex().create_index(
        seattle_dataset,
        date="sale_date",
        price="sale_price",
        trans_id="sale_id",
        prop_id="pinx",
        estimator="base",
        log_dep=True,
        periodicity="monthly",
    )
    assert full_1.model.params["estimator"] == "base"

    # Check min date with clipping.
    mindate_index = RepeatTransactionIndex().create_index(
        seattle_dataset,
        date="sale_date",
        price="sale_price",
        trans_id="sale_id",
        prop_id="pinx",
        estimator="base",
        log_dep=True,
        min_date="2011-01-01",
        adj_type="clip",
    )
    assert mindate_index.name[0] == "2011"

    # Check max date.
    maxdate_index = RepeatTransactionIndex().create_index(
        seattle_dataset,
        date="sale_date",
        price="sale_price",
        trans_id="sale_id",
        prop_id="pinx",
        estimator="base",
        log_dep=True,
        max_date="2015-12-31",
    )
    assert maxdate_index.name[6] == "2016"

    # Check periodicity.
    per_index = RepeatTransactionIndex().create_index(
        seattle_dataset,
        date="sale_date",
        price="sale_price",
        trans_id="sale_id",
        prop_id="pinx",
        estimator="base",
        log_dep=True,
        periodicity="weekly",
    )
    assert per_index.name[363] == "week: 2016-12-11 to 2016-12-17"

    # Check sequence only.
    seq_index = RepeatTransactionIndex().create_index(
        seattle_dataset,
        date="sale_date",
        price="sale_price",
        trans_id="sale_id",
        prop_id="pinx",
        estimator="base",
        log_dep=True,
        periodicity="monthly",
        seq_only=True,
    )
    assert len(seq_index.data.trans_df) == 4823


@pytest.mark.usefixtures("seattle_dataset")
@pytest.mark.parametrize("estimator", ["base", "robust", "weighted"])
@pytest.mark.parametrize("log_dep", [True, False])
def test_rt_create_trans(seattle_dataset: pd.DataFrame, estimator: str, log_dep: bool) -> None:
    """Test repeat transaction index creation."""
    repeat_trans_data = RepeatTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
    )

    model_base = RepeatTransactionModel(repeat_trans_data).fit(
        estimator=estimator,
        log_dep=log_dep,
    )

    # Check beginning value imputation.
    model_ex = copy.deepcopy(model_base)
    model_ex.coefficients.loc[1, "coefficient"] = np.nan
    assert pd.notnull(RepeatTransactionIndex.from_model(model_ex).value[1])
    assert RepeatTransactionIndex.from_model(model_ex).value[1] == 100
    assert RepeatTransactionIndex.from_model(model_ex).imputed[1] == 1

    # Check interior values interpolation.
    model_ex = copy.deepcopy(model_base)
    model_ex.coefficients.loc[2:5, "coefficient"] = np.nan
    assert np.all(pd.notnull(RepeatTransactionIndex.from_model(model_ex).value[2:5]))

    # Check end period extrapolation.
    model_ex = copy.deepcopy(model_base)
    model_ex.coefficients.loc[80:, "coefficient"] = np.nan
    model_to_index = RepeatTransactionIndex.from_model(model_ex)
    assert np.all(pd.notnull(model_to_index.value[80:]))
    assert model_to_index.value[80] == model_to_index.value[len(model_to_index.value) - 1]

    # Check shortened index.
    model_ex = copy.deepcopy(model_base)
    index_80 = RepeatTransactionIndex.from_model(model_ex, max_period=80)
    assert len(index_80.value) == 80


@pytest.mark.usefixtures("seattle_dataset")
def test_base_house_price_index_methods(seattle_dataset: pd.DataFrame) -> None:
    """Test base class methods and error handling."""

    class TestIndex(BaseHousePriceIndex):
        pass

    with pytest.raises(NotImplementedError):
        TestIndex.get_data()

    with pytest.raises(NotImplementedError):
        TestIndex.get_model()

    with pytest.raises(NotImplementedError):
        TestIndex._create_transactions(seattle_dataset)

    with pytest.raises(NotImplementedError):
        TestIndex._create_model()


@pytest.mark.usefixtures("seattle_dataset")
def test_index_creation_edge_cases(seattle_dataset: pd.DataFrame) -> None:
    """Test edge cases in index creation."""
    trans_data = RepeatTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
    )

    # Check invalid max_period type.
    with pytest.raises(ValueError, match="'max_period' argument must be numeric/integer"):
        model = RepeatTransactionModel(trans_data).fit(
            estimator="base",
            log_dep=True,
        )
        RepeatTransactionIndex.from_model(model, max_period="invalid")

    # Check failed index conversion.
    class BrokenIndex(BaseHousePriceIndex):
        @classmethod
        def from_model(cls, *args, **kwargs) -> None:
            return None


@pytest.mark.usefixtures("seattle_dataset")
def test_coef_to_index_edge_cases(seattle_dataset: pd.DataFrame) -> None:
    """Test edge cases in coefficient to index conversion."""
    # Check all coefficients missing after first.
    coef_df = pd.DataFrame({"time": range(1, 6), "coefficient": [0] + [np.nan] * 4})
    index, imputed = BaseHousePriceIndex.coef_to_index(coef_df, log_dep=True)
    assert np.all(pd.notnull(index))

    # Check all coefficients are zero or very small.
    coef_df = pd.DataFrame({"time": range(1, 6), "coefficient": [1e-16] * 5})
    index, imputed = BaseHousePriceIndex.coef_to_index(coef_df, log_dep=True)
    assert np.all(index == 100)  # All should be base value
    assert np.all(imputed == 0)  # None should be imputed since they're just small

    # Check non-log dependent variable
    coef_df = pd.DataFrame({"time": range(1, 6), "coefficient": [0.1, 0.2, 0.3, 0.4, 0.5]})
    index, imputed = BaseHousePriceIndex.coef_to_index(coef_df, log_dep=False, base_price=2)
    assert np.all(pd.notnull(index))
    assert np.all(index > 100)  # Should increase from base
    assert np.sum(imputed) == 0  # None should be imputed


@pytest.mark.usefixtures("seattle_dataset")
def test_index_creation_with_data(seattle_dataset: pd.DataFrame) -> None:
    """Test index creation with transaction data."""
    trans_data = RepeatTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
    )

    model = RepeatTransactionModel(trans_data).fit(
        estimator="base",
        log_dep=True,
    )

    # Check index with transaction data.
    index = RepeatTransactionIndex.from_model(model, trans_data=trans_data)
    assert hasattr(index, "data")
    assert index.data is not None


@pytest.mark.usefixtures("seattle_dataset")
def test_index_smoothing(seattle_dataset: pd.DataFrame) -> None:
    """Test index smoothing functionality."""
    trans_data = RepeatTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
    )

    model = RepeatTransactionModel(trans_data).fit(
        estimator="base",
        log_dep=True,
    )

    index = RepeatTransactionIndex.from_model(model)

    n = len(index.value)
    valid_order = min(3, n // 2 - 1)  # ensure order is valid

    # Check smoothing with single order.
    smoothed_index = index.smooth_index(order=valid_order, in_place=False)
    assert isinstance(smoothed_index, pd.Series)

    # Check smoothing with multiple orders.
    smoothed_index = index.smooth_index(order=[valid_order, valid_order + 1], in_place=False)
    assert isinstance(smoothed_index, pd.Series)

    # Check in-place smoothing.
    index.smooth_index(order=valid_order, in_place=True)
    assert hasattr(index, "smooth")
    assert index.smooth is not None


@pytest.mark.usefixtures("seattle_dataset")
def test_index_creation_with_smoothing(seattle_dataset: pd.DataFrame) -> None:
    """Test index creation with smoothing."""
    index = RepeatTransactionIndex().create_index(
        seattle_dataset,
        date="sale_date",
        price="sale_price",
        trans_id="sale_id",
        prop_id="pinx",
        estimator="base",
        log_dep=True,
        smooth=True,
        smooth_order=3,
    )

    assert hasattr(index, "smooth")
    assert index.smooth is not None


@pytest.mark.usefixtures("seattle_dataset")
def test_hedonic_index_methods(seattle_dataset: pd.DataFrame) -> None:
    """Test hedonic index specific methods."""
    assert HedonicIndex.get_data() == HedonicTransactionData
    assert HedonicIndex.get_model() == HedonicModel

    # Check create transactions.
    trans_data = HedonicIndex._create_transactions(
        seattle_dataset,
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
    )
    assert isinstance(trans_data, HedonicTransactionData)

    # Check create model.
    model = HedonicIndex._create_model(
        trans_data,
        dep_var="price",
        ind_var=["tot_sf"],
        estimator="base",
        log_dep=True,
    )
    assert isinstance(model, HedonicModel)


@pytest.mark.usefixtures("seattle_dataset")
def test_repeat_transaction_index_methods(seattle_dataset: pd.DataFrame) -> None:
    """Test repeat transaction index specific methods."""
    assert RepeatTransactionIndex.get_data() == RepeatTransactionData
    assert RepeatTransactionIndex.get_model() == RepeatTransactionModel

    # Check create transactions.
    trans_data = RepeatTransactionIndex._create_transactions(
        seattle_dataset,
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
    )
    assert isinstance(trans_data, RepeatTransactionData)

    # Check create model.
    model = RepeatTransactionIndex._create_model(
        trans_data,
        estimator="base",
        log_dep=True,
    )
    assert isinstance(model, RepeatTransactionModel)
