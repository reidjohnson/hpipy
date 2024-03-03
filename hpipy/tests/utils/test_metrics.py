import copy

import numpy as np
import pandas as pd
import pytest

from hpipy.period_table import PeriodTable
from hpipy.price_index import HedonicIndex, RepeatTransactionIndex
from hpipy.trans_data import HedonicTransactionData, RepeatTransactionData
from hpipy.utils.metrics import (
    accuracy,
    forecast_error,
    insample_error,
    kfold_error,
    revision,
    series_accuracy,
    series_volatility,
    volatility,
)


@pytest.mark.usefixtures("seattle_dataset")
def test_metrics(seattle_dataset: pd.DataFrame) -> None:
    # Basic sales dataframe.
    sales_df = PeriodTable(seattle_dataset).create_period_table(
        date="sale_date",
        periodicity="monthly",
    )

    # Hedonic data.
    hedonic_trans_data = HedonicTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
    )

    # Repeat sales data.
    repeat_trans_data = RepeatTransactionData(sales_df).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
    )

    # Full hedonic index.
    hed_index = HedonicIndex().create_index(
        hedonic_trans_data,
        estimator="weighted",
        log_dep=False,
        dep_var="price",
        ind_var=["tot_sf", "beds", "baths"],
        weights=np.random.uniform(0, 1, size=len(hedonic_trans_data.trans_df)),
        smooth=True,
    )

    # Full repeat sales index.
    rt_index = RepeatTransactionIndex().create_index(
        repeat_trans_data,
        estimator="base",
        log_dep=True,
        periodicity="monthly",
        smooth=True,
    )

    assert hed_index
    assert rt_index

    hed_series = hed_index.create_series(train_period=24)
    assert isinstance(hed_series, HedonicIndex)

    rt_series = rt_index.create_series(train_period=24, max_period=50)
    assert isinstance(rt_series, RepeatTransactionIndex)
    assert len(rt_series.hpis) == 27

    # Max period is limited to length of HPI object index.
    hed_series = hed_index.create_series(train_period=12, max_period=150)
    assert len(hed_series.hpis) == 73

    # Bad HPI object.
    with pytest.raises(AttributeError):
        _ = hed_index.index.create_series(train_period=24, max_period=50)

    # Bad train period.
    with pytest.raises(ValueError):
        _ = hed_index.create_series(train_period="x", max_period=50)  # type: ignore

    # Bad train period.
    with pytest.raises(ValueError):
        _ = hed_index.create_series(train_period=99, max_period=50)

    # Create series.
    rt_series = rt_index.create_series(train_period=24)

    # Standard return.
    rt_series = rt_series.smooth_series(order=5)
    assert isinstance(rt_series, RepeatTransactionIndex)
    for hpi in rt_series.hpis:
        assert hasattr(hpi, "smooth")

    # Bad series object.
    with pytest.raises(AttributeError):
        _ = rt_series.hpis[0].smooth_series(order=5)

    # Bad order.
    with pytest.raises(ValueError):
        _ = rt_series.smooth_series(order=-1)

    # Create series for remaining analyses.
    hed_series = hed_index.create_series(train_period=24, max_period=30)
    rt_series = rt_index.create_series(train_period=24)
    rt_series = rt_series.smooth_series()

    _ = volatility(hed_index.value, window=3)  # type: ignore

    # HPI index object.
    _ = volatility(hed_index, window=3)

    # Full HPI object.
    _ = volatility(hed_index, window=3)

    # Standard input.
    _ = volatility(hed_index.smooth, window=3)

    # HPI index object.
    _ = volatility(hed_index, window=3, smooth=True)

    # Throws error if smooth is gone.
    ex_index = copy.deepcopy(hed_index)
    ex_index.smooth = None

    with pytest.raises(ValueError):
        _ = volatility(ex_index, window=3, smooth=True)

    with pytest.raises(AttributeError):
        _ = volatility(ex_index.index, window=3, smooth=True)

    # Full HPI object.
    _ = volatility(hed_index, window=3, smooth=True)

    # Non-sensical index.
    with pytest.raises(ValueError):
        _ = volatility("abc", window=3)  # type: ignore

    # Negative window.
    with pytest.raises(ValueError):
        _ = volatility(hed_index, window=-1)

    # Char window.
    with pytest.raises(ValueError):
        _ = volatility(hed_index, window="x")  # type: ignore

    # NA window.
    with pytest.raises(ValueError):
        _ = volatility(hed_index, window=None, smooth=True)  # type: ignore

    # Standard input.
    _ = volatility(hed_index.value, window=3, in_place=True)  # type: ignore

    # Add it to the HPI index object.
    hed_index.index = volatility(hed_index, window=3, in_place=True)

    # Add it to the HPI index object smooth.
    hed_index.index = volatility(hed_index, window=3, in_place=True, smooth=True)

    # Add it to the full HPI object (to the HPI index object).
    hed_index = volatility(hed_index, window=3, in_place=True)  # type: ignore

    # Add it to the full HPI object (to the hpiindex object) smooth.
    hed_index = volatility(hed_index, window=3, in_place=True, smooth=True)  # type: ignore

    # Add it to the full HPI object (to the hpiindex object) with new name.
    # hed_index = volatility(hed_index, window=3, in_place=True, in_place_name="xxx")

    # Standard input.
    _ = series_volatility(rt_series, window=3)

    _ = series_volatility(rt_series, window=3, smooth=True)

    # Bad HPI objects.
    with pytest.raises(ValueError):
        accuracy("xxx")  # type: ignore
    with pytest.raises(ValueError):
        accuracy(hed_index.data)  # type: ignore

    # Disagreement between HPI object and prediction dataframe.
    # with pytest.raises(ValueError):
    #     accuracy(hed_index, test_type="rt")
    # with pytest.raises(ValueError):
    #     accuracy(rt_index, test_type="hed")
    # with pytest.raises(ValueError):
    #     accuracy(hed_index, test_type="rt", pred_df=hed_index.data)
    # with pytest.raises(ValueError):
    #     accuracy(rt_index, test_type="hed", pred_df=rt_index.data)

    # Bad test_method.
    with pytest.raises(ValueError):
        accuracy(rt_index, test_type="rt", test_method="x")

    # Bad test type.
    with pytest.raises(ValueError):
        accuracy(rt_index, test_type="x", test_method="insample")

    # Bad data.
    with pytest.raises(ValueError):
        _ = insample_error(
            hed_index.data.trans_df, hed_index, index=hed_index.value  # type: ignore
        )

    # Bad index.
    with pytest.raises(ValueError):
        _ = insample_error(hed_index.data.trans_df, hed_index, index=hed_index)  # type: ignore

    # All data.
    with pytest.raises(ValueError):
        _ = insample_error(
            hed_index.data.trans_df, hed_index, index=hed_index.value  # type: ignore
        )

    # All data smooth.
    with pytest.raises(ValueError):
        _ = insample_error(
            hed_index.data.trans_df, hed_index, index=hed_index.smooth  # type: ignore
        )

    # Sparse data.
    with pytest.raises(ValueError):
        _ = insample_error(
            hed_index.data.trans_df.iloc[:3], hed_index, index=hed_index.value  # type: ignore
        )

    # No data.
    with pytest.raises(ValueError):
        _ = insample_error(
            hed_index.data.trans_df.iloc[:1], hed_index, index=hed_index.value  # type: ignore
        )

    # Bad HPI object.
    with pytest.raises(ValueError):
        _ = kfold_error(rt_index, pred_df=rt_index.data)  # type: ignore

    # Bad prediction dataframe.
    with pytest.raises(ValueError):
        _ = kfold_error(rt_index, pred_df=rt_index)  # type: ignore

    # Bad k.
    with pytest.raises(ValueError):
        _ = kfold_error(rt_index, pred_df=rt_index.data, k="a")  # type: ignore

    # Bad seed.
    with pytest.raises(ValueError):
        _ = kfold_error(rt_index, pred_df=rt_index.data, seed="x")  # type: ignore

    # All data.
    with pytest.raises(ValueError):
        rt_error = kfold_error(rt_index, pred_df=rt_index.data)  # type: ignore
        assert len(rt_error.columns) == 6

    # All data - smooth.
    with pytest.raises(ValueError):
        _ = kfold_error(rt_index, pred_df=rt_index.data, smooth=True)  # type: ignore

    # Sparse data.
    # with pytest.raises(ValueError):
    #     _ = kfold_error(rt_index, pred_df=rt_index.data.trans_df.iloc[0:39])

    # No data.
    with pytest.raises(ValueError):
        _ = kfold_error(rt_index, pred_df=rt_index.data.trans_df.iloc[0])  # type: ignore

    # Returns an error.
    # with pytest.raises(ValueError):
    #     rt_error = accuracy(
    #         rt_index, test_type="rt", test_method="insample", pred_df=rt_index.data
    #     )
    #     assert len(rt_error.columns) == 6

    # Returns an error.
    with pytest.raises(ValueError):
        rt_index = accuracy(  # type: ignore
            rt_index,
            test_type="rt",
            test_method="insample",
            pred_df=rt_index.data.trans_df,
            in_place=True,
            in_place_name="acc",
        )
        assert rt_index.acc  # type: ignore

    # Returns an error.
    # with pytest.raises(ValueError):
    #     rt_error = accuracy(rt_index, test_type="rt", test_method="kfold", pred_df=rt_index.data)
    #     assert len(rt_error.columns) == 6

    # Returns an error.
    with pytest.raises(ValueError):
        rt_index = accuracy(  # type: ignore
            hpi_obj=rt_index,
            test_type="rt",
            test_method="kfold",
            pred_df=rt_index.data,
            in_place=True,
            in_place_name="errors",
        )
        assert rt_index.errors  # type: ignore

    # Bad series.
    with pytest.raises(AttributeError):
        _ = series_accuracy(rt_series.data, test_method="insample", test_type="rt")  # type: ignore

    # Bad test method.
    with pytest.raises(ValueError):
        _ = series_accuracy(rt_series, test_method="xxx", test_type="rt", smooth=True)

    # Bad test type.
    with pytest.raises(ValueError):
        _ = series_accuracy(rt_series, test_method="kfold", test_type="rtx", smooth=True)

    # Bad prediction dataframe.
    with pytest.raises(ValueError):
        _ = hed_series = series_accuracy(
            hed_series, test_method="insample", test_type="rt"  # type: ignore
        )

    # Smooth and in place.
    rt_series = series_accuracy(
        rt_series,
        test_method="insample",
        test_type="rt",
        smooth=True,
        in_place=True,  # type: ignore
    )

    # Smooth when no smooth existing.
    with pytest.raises(ValueError):
        _ = series_accuracy(
            hed_series, test_method="insample", test_type="rt", smooth=True, pred_df=rt_series.data
        )

    # Smooth when no smooth existing.
    with pytest.raises(ValueError):
        _ = series_accuracy(
            series_obj=hed_series,
            test_method="kfold",
            test_type="rt",
            smooth=True,
            pred_df=rt_series.data,
        )

    series_accuracy(
        series_obj=rt_series, test_method="insample", test_type="rt", summarize=True, in_place=True
    )

    _ = hed_index.data.create_forecast_periods(time_cut=33, train=True)

    _ = rt_index.data.create_forecast_periods(time_cut=33, train=True)

    _ = hed_index.data.create_forecast_periods(time_cut=33, forecast_length=2, train=True)

    _ = rt_index.data.create_forecast_periods(time_cut=33, train=True)

    # Bad data.
    with pytest.raises(AttributeError):
        _ = hed_index.create_forecast_periods(time_cut=33, train=True)  # type: ignore

    # Bad time cut.
    with pytest.raises(ValueError):
        _ = hed_index.data.create_forecast_periods(time_cut=-1, train=True)

    # Bad forecast length.
    with pytest.raises(ValueError):
        _ = hed_index.data.create_forecast_periods(time_cut=33, forecast_length="x", train=True)

    # Bad series object.
    with pytest.raises(ValueError):
        _ = rt_index.data.create_forecast_periods(hed_index)

    # Bad prediction dataframe.
    with pytest.raises(ValueError):
        _ = rt_index.data.create_forecast_periods(hed_series)

    # Smooth when not present.
    with pytest.raises(AttributeError):
        _ = forecast_error(hed_series, trans_data=rt_index.data, smooth=True)

    # Smooth when not present.
    with pytest.raises(ValueError):
        _ = forecast_error(
            hed_series, trans_data=rt_index.data, forecast_length="x"  # type: ignore
        )

    # All data.
    # hed_acc = forecast_error(hed_series, trans_data=rt_index.data)

    # All data, longer forecast length.
    # hed_acc = forecast_error(hed_series, trans_data=rt_index.data, forecast_length=3)

    # All data, smoothed.
    _ = forecast_error(rt_series, trans_data=rt_index.data, smooth=True)

    # Sparse data.
    trans_data = copy.deepcopy(rt_index.data)
    trans_data.trans_df = trans_data.trans_df.iloc[:39]
    _ = forecast_error(rt_series, trans_data=trans_data)

    # No data.
    trans_data = copy.deepcopy(rt_index.data)
    trans_data.trans_df = trans_data.trans_df.iloc[:1]
    _ = forecast_error(rt_series, trans_data=trans_data, smooth=True)

    # Returns a series with accuracy: smooth and in place.
    rt_series = series_accuracy(  # type: ignore
        series_obj=rt_series,
        test_type="rt",
        test_method="forecast",
        pred_df=rt_series.data,
        smooth=True,
        in_place=True,
    )

    # Standard series object.
    _ = revision(series_obj=hed_series)

    # In place.
    hed_series = revision(series_obj=hed_series, in_place=True)  # type: ignore

    # With smooth.
    _ = revision(series_obj=rt_series, smooth=True)

    # With smooth in place.
    rt_series = revision(series_obj=rt_series, smooth=True, in_place=True)  # type: ignore

    # Bad series object.
    with pytest.raises(ValueError):
        _ = revision(series_obj=hed_series.data)  # type: ignore

    # Bad smooth = True.
    with pytest.raises(AttributeError):
        _ = revision(series_obj=hed_series, smooth=True)
