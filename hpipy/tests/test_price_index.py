import copy

import numpy as np
import pandas as pd
import pytest

from hpipy.price_index import RepeatTransactionIndex
from hpipy.price_model import RepeatTransactionModel
from hpipy.trans_data import RepeatTransactionData


@pytest.mark.usefixtures("seattle_dataset")
def test_rt_create_trans_toy(seattle_dataset: pd.DataFrame) -> None:
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


@pytest.mark.usefixtures("seattle_dataset")  # type: ignore
@pytest.mark.parametrize("estimator", ["base", "robust", "weighted"])
@pytest.mark.parametrize("log_dep", [True, False])
def test_rt_create_trans(seattle_dataset: pd.DataFrame, estimator: str, log_dep: bool) -> None:
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
