import warnings
from typing import Optional

import pandas as pd
import pytest

from hpipy.extensions import RandomForestIndex


@pytest.mark.usefixtures("seattle_dataset")
@pytest.mark.parametrize("estimator", ["pdp"])
@pytest.mark.parametrize("log_dep", [True, False])
def test_rf_create_trans(seattle_dataset: pd.DataFrame, estimator: str, log_dep: bool) -> None:
    rf_index = RandomForestIndex().create_index(
        seattle_dataset,
        date="sale_date",
        price="sale_price",
        trans_id="sale_id",
        prop_id="pinx",
        dep_var="price",
        ind_var=["tot_sf", "beds", "baths"],
        estimator=estimator,
        log_dep=log_dep,
        periodicity="monthly",
    )
    assert rf_index.model.params["estimator"] == estimator


@pytest.mark.usefixtures("seattle_dataset")
@pytest.mark.parametrize("log_dep", [True, False])
@pytest.mark.parametrize("sim_ids", [None, [0, 1]])
@pytest.mark.parametrize("sim_count", [None, 10])
@pytest.mark.parametrize("sim_per", [None, 0.1])
def test_rf_create_trans_small(
    seattle_dataset: pd.DataFrame,
    log_dep: bool,
    sim_ids: Optional[list[int]],
    sim_count: Optional[int],
    sim_per: Optional[float],
) -> None:
    rf_index = RandomForestIndex().create_index(
        seattle_dataset.iloc[:100],
        date="sale_date",
        price="sale_price",
        trans_id="sale_id",
        prop_id="pinx",
        dep_var="price",
        ind_var=["tot_sf", "beds", "baths"],
        estimator="pdp",
        log_dep=log_dep,
        periodicity="monthly",
        sim_ids=sim_ids,
        sim_count=sim_count,
        sim_per=sim_per,
    )
    assert rf_index.model.params["estimator"] == "pdp"


@pytest.mark.usefixtures("seattle_dataset")
def test_rf_create_trans_bad(seattle_dataset: pd.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with pytest.raises(ValueError):
            _ = RandomForestIndex().create_index(
                seattle_dataset.assign(
                    sale_price=lambda x: x["sale_price"] - x["sale_price"].min() - 1
                ),
                date="sale_date",
                price="sale_price",
                trans_id="sale_id",
                prop_id="pinx",
                dep_var="price",
                ind_var=["tot_sf", "beds", "baths"],
                estimator="pdp",
                log_dep=True,
                periodicity="monthly",
            )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with pytest.raises(ValueError):
            _ = RandomForestIndex().create_index(
                seattle_dataset.assign(
                    sale_price=lambda x: x["sale_price"] - x["sale_price"].min() - 1
                ),
                date="sale_date",
                price="sale_price",
                trans_id="sale_id",
                prop_id="pinx",
                dep_var="price",
                ind_var=["tot_sf", "beds", "baths"],
                estimator="x",
                log_dep=True,
                periodicity="monthly",
            )
