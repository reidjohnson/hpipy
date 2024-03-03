import copy
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import HuberRegressor, LinearRegression

from hpipy.price_index import HedonicIndex
from hpipy.price_model import HedonicModel, RepeatTransactionModel
from hpipy.time_matrix import TimeMatrixMixin
from hpipy.trans_data import HedonicTransactionData, RepeatTransactionData


@pytest.mark.usefixtures("seattle_dataset")
def test_rt_model(seattle_dataset: pd.DataFrame) -> None:
    repeat_trans_data = RepeatTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
    )

    # Check negative price.
    repeat_trans_datax = copy.deepcopy(repeat_trans_data)
    repeat_trans_datax.trans_df.loc[0, "price_1"] = -1
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with pytest.raises(ValueError):
            RepeatTransactionModel(repeat_trans_datax).fit(estimator="base", log_dep=True)

    # Check NaN price.
    repeat_trans_datax = copy.deepcopy(repeat_trans_data)
    repeat_trans_datax.trans_df.loc[0, "price_1"] = np.nan
    with pytest.raises(ValueError):
        RepeatTransactionModel(repeat_trans_datax).fit(estimator="base", log_dep=True)

    rt_model = RepeatTransactionModel(repeat_trans_data).fit()
    assert rt_model.params["estimator"] == "base"

    rt_model = RepeatTransactionModel(repeat_trans_data).fit(estimator="robust")
    assert rt_model.params["estimator"] == "robust"

    rt_model = RepeatTransactionModel(repeat_trans_data).fit(
        hpi_data=repeat_trans_data, estimator="weighted"
    )
    assert rt_model.params["estimator"] == "weighted"

    rt_model = RepeatTransactionModel(repeat_trans_data).fit(
        hpi_data=repeat_trans_data, estimator="x"
    )
    assert rt_model.params["estimator"] == "base"

    repeat_trans_data200 = copy.deepcopy(repeat_trans_data)
    repeat_trans_data200.trans_df = repeat_trans_data200.trans_df[0:199]
    time_matrix200 = TimeMatrixMixin().create_time_matrix(repeat_trans_data200.trans_df)
    price_diff_l200 = np.log1p(repeat_trans_data200.trans_df["price_2"]) - np.log1p(
        repeat_trans_data200.trans_df["price_1"]
    )
    price_diff200 = (
        repeat_trans_data200.trans_df["price_2"] - repeat_trans_data200.trans_df["price_1"]
    )

    rt_model = RepeatTransactionModel(repeat_trans_data200)._create_model(
        repeat_trans_data200.trans_df,
        repeat_trans_data200.period_table,
        time_matrix200,
        price_diff_l200,
        "base",
    )
    assert isinstance(rt_model, LinearRegression)

    rt_model = RepeatTransactionModel(repeat_trans_data200)._create_model(
        repeat_trans_data200.trans_df,
        repeat_trans_data200.period_table,
        time_matrix200,
        price_diff_l200,
        "robust",
    )
    assert isinstance(rt_model, HuberRegressor)

    rt_model = RepeatTransactionModel(repeat_trans_data200)._create_model(
        repeat_trans_data200.trans_df,
        repeat_trans_data200.period_table,
        time_matrix200,
        price_diff200,
        "weighted",
    )
    assert isinstance(rt_model, LinearRegression)

    rt_model = RepeatTransactionModel(repeat_trans_data200).fit(estimator="base")
    assert isinstance(rt_model.model_obj, LinearRegression)

    rt_model = RepeatTransactionModel(repeat_trans_data200).fit(estimator="robust")
    assert isinstance(rt_model.model_obj, HuberRegressor)

    rt_model = RepeatTransactionModel(repeat_trans_data200).fit(estimator="weighted")
    assert isinstance(rt_model.model_obj, LinearRegression)

    rt_model_base = RepeatTransactionModel(repeat_trans_data).fit(estimator="base", log_dep=True)

    rt_model_robust = RepeatTransactionModel(repeat_trans_data).fit(
        estimator="robust", log_dep=True
    )

    rt_model_wgt = RepeatTransactionModel(repeat_trans_data).fit(
        estimator="weighted", log_dep=False
    )

    np.random.seed(0)
    rt_model_wwgt = RepeatTransactionModel(repeat_trans_data).fit(
        estimator="weighted",
        log_dep=False,
        weights=np.random.uniform(0, 1, size=len(repeat_trans_data.trans_df)),
    )

    # Check estimator.
    assert rt_model_base.params["estimator"] == "base"
    assert rt_model_robust.params["estimator"] == "robust"
    assert rt_model_wgt.params["estimator"] == "weighted"

    # Check coefficients.
    assert isinstance(rt_model_base.coefficients, pd.DataFrame)
    assert isinstance(rt_model_robust.coefficients, pd.DataFrame)
    assert isinstance(rt_model_wgt.coefficients, pd.DataFrame)
    assert len(rt_model_base.coefficients) == 84
    assert len(rt_model_robust.coefficients) == 84
    assert rt_model_wgt.coefficients.loc[0, "coefficient"] == 0
    assert np.any(
        np.not_equal(
            rt_model_wgt.coefficients["coefficient"],
            rt_model_wwgt.coefficients["coefficient"],
        )
    )

    # Check base price.
    assert np.round(rt_model_base.base_price) == 427785
    assert np.round(rt_model_robust.base_price) == 427785
    assert np.round(rt_model_wgt.base_price) == 427785

    # Check periods.
    assert len(rt_model_base.periods) == 84
    assert len(rt_model_robust.periods) == 84
    assert len(rt_model_wgt.periods) == 84


@pytest.mark.usefixtures("seattle_dataset")
def test_hed_model(seattle_dataset: pd.DataFrame) -> None:
    hedonic_trans_data = HedonicTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
    )

    hed_model_base = HedonicModel(hedonic_trans_data).fit(
        estimator="base",
        dep_var="price",
        ind_var=["tot_sf", "beds", "baths"],
        log_dep=True,
    )

    hed_model_robust = HedonicModel(hedonic_trans_data).fit(
        estimator="robust",
        dep_var="price",
        ind_var=["tot_sf", "beds", "baths"],
        log_dep=True,
    )

    hed_model_wgt = HedonicModel(hedonic_trans_data).fit(
        estimator="weighted",
        dep_var="price",
        ind_var=["tot_sf", "beds", "baths"],
        log_dep=False,
        weights=np.random.uniform(0, 1, size=len(hedonic_trans_data.trans_df)),
    )

    hed_model_x = HedonicModel(hedonic_trans_data).fit(
        estimator="x",
        dep_var="price",
        ind_var=["tot_sf", "beds", "baths"],
    )

    # Check estimator.
    assert hed_model_base.params["estimator"] == "base"
    assert hed_model_robust.params["estimator"] == "robust"
    assert hed_model_wgt.params["estimator"] == "weighted"
    assert hed_model_x.params["estimator"] == "base"

    # Check coefficients.
    assert isinstance(hed_model_base.coefficients, pd.DataFrame)
    assert isinstance(hed_model_robust.coefficients, pd.DataFrame)
    assert isinstance(hed_model_wgt.coefficients, pd.DataFrame)
    assert len(hed_model_base.coefficients) == 84
    assert hed_model_robust.coefficients["time"].max() == 84
    assert hed_model_wgt.coefficients.loc[0, "coefficient"] == 0

    # Check base price.
    assert np.round(hed_model_base.base_price) == 462545
    assert np.round(hed_model_robust.base_price) == 462545
    assert np.round(hed_model_wgt.base_price) == 462545

    # Check periods.
    assert isinstance(hed_model_base.periods, pd.DataFrame)
    assert len(hed_model_base.periods) == 84
    assert isinstance(hed_model_robust.periods, pd.DataFrame)
    assert len(hed_model_robust.periods) == 84
    assert isinstance(hed_model_wgt.periods, pd.DataFrame)
    assert len(hed_model_wgt.periods) == 84

    model_base = HedonicModel(hedonic_trans_data).fit(
        estimator="base",
        log_dep=True,
        dep_var="price",
        ind_var=["tot_sf", "beds", "baths"],
    )

    # Check impute a beginning value.
    model_ex = copy.deepcopy(model_base)
    model_ex.coefficients.loc[1, "coefficient"] = np.nan
    assert pd.notnull(HedonicIndex.from_model(model_ex).value[2])
    assert HedonicIndex.from_model(model_ex).value[1] == 100
    assert HedonicIndex.from_model(model_ex).imputed[1] == 1

    # Check interpolate interior values.
    model_ex = copy.deepcopy(model_base)
    model_ex.coefficients.loc[2:5, "coefficient"] = np.nan
    assert np.all(pd.notnull(HedonicIndex.from_model(model_ex).value[3:5]))

    # Check extrapolate end periods.
    model_ex = copy.deepcopy(model_base)
    model_ex.coefficients.loc[80:84, "coefficient"] = np.nan
    assert np.all(pd.notnull(HedonicIndex.from_model(model_ex).value[81:84]))
    assert (
        HedonicIndex.from_model(model_ex).value[79] == HedonicIndex.from_model(model_ex).value[83]
    )

    model_base = HedonicModel(hedonic_trans_data).fit(
        estimator="base",
        log_dep=False,
        dep_var="price",
        ind_var=["tot_sf", "beds", "baths"],
    )

    # Check extrapolate a beginning value.
    model_ex = copy.deepcopy(model_base)
    model_ex.coefficients.loc[1, "coefficient"] = np.nan
    assert pd.notnull(HedonicIndex.from_model(model_ex).value[2])
    assert HedonicIndex.from_model(model_ex).value[1] == 100
    assert HedonicIndex.from_model(model_ex).imputed[1] == 1

    # Check impute interior values.
    model_ex = copy.deepcopy(model_base)
    model_ex.coefficients.loc[2:5, "coefficient"] = np.nan
    assert np.all(pd.notnull(HedonicIndex.from_model(model_ex).value[3:5]))

    # Check extrapolate an end value.
    model_ex = copy.deepcopy(model_base)
    model_ex.coefficients.loc[80:84, "coefficient"] = np.nan
    assert np.all(pd.notnull(HedonicIndex.from_model(model_ex).value[81:84]))
    assert (
        HedonicIndex.from_model(model_ex).value[79] == HedonicIndex.from_model(model_ex).value[83]
    )

    # Check minimum date with clipping.
    addarg_index = HedonicIndex().create_index(
        seattle_dataset,
        date="sale_date",
        price="sale_price",
        trans_id="sale_id",
        prop_id="pinx",
        estimator="robust",
        dep_var="price",
        ind_var=["tot_sf", "beds", "baths"],
        min_date="2011-01-01",
        max_date="2015-12-31",
        periodicity="annual",
        adj_type="clip",
    )
    assert addarg_index.name[4] == "2015"

    # Check log dep and robust.
    ld_index = HedonicIndex().create_index(
        hedonic_trans_data,
        estimator="robust",
        dep_var="price",
        ind_var=["tot_sf", "beds", "baths"],
        log_dep=False,
        smooth=True,
        smooth_order=5,
    )
    assert ld_index.model.params["log_dep"] is False
    assert ld_index.model.params["estimator"] == "robust"

    # Check model to index.
    m2i_index = HedonicIndex().create_index(
        hedonic_trans_data,
        estimator="robust",
        log_dep=False,
        dep_var="price",
        ind_var=["tot_sf", "beds", "baths"],
        max_period=80,
    )
    assert len(m2i_index.value) == 80
