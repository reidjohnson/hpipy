"""Metrics utilities."""

import copy
import logging
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from hpipy.price_index import BaseHousePriceIndex, HedonicIndex, RepeatTransactionIndex
from hpipy.price_model import BaseHousePriceModel, HedonicModel, RepeatTransactionModel
from hpipy.trans_data import HedonicTransactionData, RepeatTransactionData, TransactionData


def accuracy(
    hpi_obj: BaseHousePriceIndex,
    test_method: str = "insample",
    test_type: str = "rt",
    pred_df: Optional[Union[TransactionData, pd.DataFrame]] = None,
    smooth: bool = False,
    in_place: bool = False,
    in_place_name: str = "accuracy",
    **kwargs: Any,
) -> Union[BaseHousePriceIndex, pd.DataFrame]:
    """Calculate the accuracy of an index.

    Args:
        hpi_obj (BaseHousePriceIndex): Index object.
        test_method (str, optional): Testing method.
            Defaults to "insample".
        test_type (str, optional): Testing type.
            Defaults to "rt".
        smooth (bool, optional): Smooth the index. If True, the revision is
            calculated based on the smoothed indices.
            Defaults to False.
        in_place (bool, optional): Return accuracy in-place.
            Defaults to False.

    Returns:
        Index object containing the accuracy or DataFrame.
    """
    # Check for class of hpi_obj.
    if not isinstance(hpi_obj, BaseHousePriceIndex):
        raise ValueError("'hpi_obj' must be an index instance.")

    # Check for allowed test_method.
    if test_method not in ["insample", "kfold"]:
        raise ValueError("'test_method' must be one of 'insample' or 'kfold'.")

    # Check for allowed test_method.
    if test_type not in ["rt", "hed"]:
        raise ValueError("'test_type' must be one of 'rt', 'hed'.")

    # Check agreement between test_type and hpi_obj
    if isinstance(pred_df, pd.DataFrame):
        df: pd.DataFrame = pred_df
    else:
        df = hpi_obj.data.trans_df
    index_type = hpi_obj.model

    # Check for smooth.
    if smooth and not hasattr(hpi_obj, "smooth"):
        raise ValueError(
            "'hpi_obj' has no smoothed index. Please add one or set 'smooth' to False."
        )

    # Clip pred_df to size of index.
    if test_type == "rt":
        if df["period_2"].max() > hpi_obj.periods.max():
            logging.info(
                f"Trimming prediction date down to period {hpi_obj.periods.max()} and before."
            )
            df = df[df["period_2"] <= hpi_obj.periods.max()]

    # In-sample.
    if test_method == "insample":
        index = hpi_obj.value
        if smooth:
            index = hpi_obj.smooth
        accr_obj = insample_error(pred_df=df, index_type=index_type, index=index)

    # K-fold.
    if test_method == "kfold":
        accr_obj = kfold_error(hpi_obj=hpi_obj, pred_df=df, smooth=smooth, **kwargs)

    if in_place:
        setattr(hpi_obj, in_place_name, accr_obj)
        return hpi_obj

    return accr_obj


def series_accuracy(
    series_obj: BaseHousePriceIndex,
    test_method: str = "insample",
    test_type: str = "rt",
    pred_df: Optional[Union[TransactionData, pd.DataFrame]] = None,
    smooth: bool = False,
    summarize: bool = False,
    in_place: bool = False,
    in_place_name: str = "accuracy",
    **kwargs: Any,
) -> Union[BaseHousePriceIndex, pd.DataFrame, list[pd.DataFrame]]:
    """Calculate the accuracy of a series.

    Args:
        series_obj (BaseHousePriceIndex): Series object.
        test_method (str, optional): Testing method.
            Defaults to "insample".
        test_type (str, optional): Testing type.
            Defaults to "rt".
        smooth (bool, optional): Smooth the index. If True, the revision is
            calculated based on the smoothed indices.
            Defaults to False.
        summarize (bool, optional): Summarize the accuracy.
            Defaults to False.
        in_place (bool, optional): Return accuracy in-place.
            Defaults to False.
        in_place_name (str, optional): In-place attribute name.
            Defualts to "accuracy".

    Returns:
        Series object containing the accuracy or DataFrame.
    """
    # Check for allowed test_method.
    if test_method not in ["insample", "kfold", "forecast"]:
        raise ValueError("'test_method' must be one of 'insample', 'kfold' or 'forecast'.")

    # Check for allowed test_method.
    if test_type not in ["rt", "hed"]:
        raise ValueError("'test_type' must be one of 'rt', 'hed'.")

    # Check agreement between test_type and hpi_obj.
    if not (
        (isinstance(series_obj.data, RepeatTransactionData) and test_type == "rt")
        or (isinstance(series_obj.data, HedonicTransactionData) and test_type == "hed")
    ):
        if pred_df is None:
            raise ValueError(
                f"When 'test_type' ({test_type}) does not match the 'hpi' object model type "
                f"({type(series_obj)}) you must provide an 'pred_df' object of the necessary "
                f"class, in this case: {test_type}."
            )
    else:
        trans_data: TransactionData = series_obj.data

    accr_dfs = []
    if test_method != "forecast":
        # Check for smooth indices.
        if smooth and not hasattr(series_obj.hpis[0], "smooth"):
            raise ValueError("No smoothed indices found. Please add or set smooth to False.")

        # Calculate accuracy.
        for idx, hpi_obj in enumerate(series_obj.hpis):
            hpi_obj_i = copy.deepcopy(series_obj)
            hpi_obj_i.data = hpi_obj.data
            hpi_obj_i.value = hpi_obj.value
            if smooth:
                hpi_obj_i.smooth = hpi_obj.smooth

            accr_df_i = pd.DataFrame(
                accuracy(
                    hpi_obj_i,
                    test_method,
                    test_type,
                    trans_data,
                    smooth=smooth,
                    in_place=False,
                )
            )
            accr_dfs.append(accr_df_i)

        accr_df = pd.concat(accr_dfs, ignore_index=True)

        # If summarizing.
        if summarize:
            accr_df = (
                accr_df.groupby("pair_id")
                .agg({"pred_price": "mean", "error": "mean", "log_error": "mean"})
                .reset_index()
            )

    # If it is a forecast method.
    else:
        accr_df = forecast_error(is_obj=series_obj, trans_data=trans_data, smooth=smooth, **kwargs)

    # Return if not in place.
    if not in_place:
        return accr_df

    # Add to series object.
    if smooth and in_place_name == "accuracy":
        in_place_name = "accuracy_smooth"
    setattr(series_obj, in_place_name, accr_df)

    return series_obj


def insample_error(
    pred_df: pd.DataFrame,
    index_type: BaseHousePriceModel,
    index: pd.Series,
) -> pd.DataFrame:
    """Calculate in-sample error."""
    if not isinstance(pred_df, pd.DataFrame):
        raise ValueError("'pred_df' argument must be a dataframe.")
    if isinstance(index_type, RepeatTransactionModel):
        return insample_error_rtdata(pred_df, index)
    elif isinstance(index_type, HedonicModel):
        return insample_error_heddata(pred_df, index)
    else:
        raise ValueError


def insample_error_rtdata(pred_df: pd.DataFrame, index: pd.Series) -> pd.DataFrame:
    """Calculate in-sample error for repeat transaction index and data."""
    # Calculate the index adjustment to apply.
    adj = []
    for idx2, idx1 in zip(pred_df["period_2"] - 1, pred_df["period_1"] - 1):
        adj.append(index.iloc[int(idx2)] / index.iloc[int(idx1)])

    # Calculate a prediction price.
    pred_price = pred_df["price_1"] * adj

    # Measure the error (difference from actual).
    error = (pred_price - pred_df["price_2"]) / pred_df["price_2"]
    logerror = np.log(pred_price) - np.log(pred_df["price_2"])

    error_df = pd.DataFrame(
        {
            "pair_id": pred_df["pair_id"],
            "rt_price": pred_df["price_2"],
            "pred_price": pred_price,
            "error": error,
            "log_error": logerror,
            "pred_period": pred_df["period_2"],
        }
    )

    return error_df


def insample_error_heddata(pred_df: pd.DataFrame, index: pd.Series, **kwargs: Any) -> pd.DataFrame:
    """Calculate in-sample error for hedonic index and data."""
    # Future method.
    raise NotImplementedError


def kfold_error(
    hpi_obj: BaseHousePriceIndex,
    pred_df: pd.DataFrame,
    smooth: bool = False,
    k: int = 10,
    seed: int = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Calculate k-fold error for an index.

    Args:
        hpi_obj (BaseHousePriceIndex): Index object.
        pred_df (pd.DataFrame): Prediction data.
        smooth (bool, optional): Smooth the index.
            Defaults to  False.
        k (int, optional): Number of folds.
            Defaults to 10.
        seed (int, optional): Seed.
            Defualts to 0.

    Returns:
        pd.DataFrame: k-fold error.
    """
    # Check hpi_obj.
    if not isinstance(hpi_obj, (HedonicIndex, RepeatTransactionIndex)):
        raise ValueError("hpi_obj argument must be of class 'hpi'.")

    # Check pred_df.
    if not isinstance(pred_df, pd.DataFrame):
        raise ValueError("'pred_df' argument must be a DataFrame.")

    # Check k.
    if not isinstance(k, (int, float)) or k < 2:
        raise ValueError(
            "Number of folds ('k' argument) must be a positive integer greater than 1."
        )

    # Check seed.
    if not isinstance(seed, (int, float)) or seed < 0:
        raise ValueError("'seed' must be a non-negative integer.")

    errors = []

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for _, test_idx in kf.split(hpi_obj.data.trans_df):
        train_df, score_df = create_kfold_data(test_idx, hpi_obj.data, pred_df)

        train_data = copy.deepcopy(hpi_obj.data)
        train_data.trans_df = train_df
        score_data = copy.deepcopy(hpi_obj.data)
        score_data.trans_df = score_df

        k_model = hpi_obj._create_model(
            train_data,
            log_dep=hpi_obj.model.params["log_dep"],
            estimator=hpi_obj.model.params["estimator"],
            **kwargs,
        )

        k_index = hpi_obj.from_model(k_model, max_period=hpi_obj.data.trans_df["period_2"].max())

        # Deal with smoothing.
        if not smooth:
            index = k_index.value
        else:
            smooth_order = 3
            if "smooth_order" in kwargs.keys():
                smooth_order = kwargs["smooth_order"]
            index = k_index.smooth_index(order=smooth_order)

        # Calc errors.
        k_error = insample_error(score_df, index_type=hpi_obj.model, index=index)
        errors.append(k_error)

    accr_df = pd.concat(errors)

    return accr_df


def create_kfold_data(
    score_ids: np.ndarray,
    full_data: TransactionData,
    pred_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create k-fold for transaction data."""
    if isinstance(full_data, RepeatTransactionData):
        return create_kfold_data_rtdata(score_ids, full_data, pred_df)
    raise NotImplementedError


def create_kfold_data_rtdata(
    score_ids: np.ndarray,
    full_data: TransactionData,
    pred_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create k-fold for repeat transaction data."""
    train_df = full_data.trans_df.iloc[
        ~full_data.trans_df.reset_index(drop=True).index.isin(score_ids)
    ]
    score_df = match_kfold(train_df, pred_df, full_data)
    return train_df, score_df


def match_kfold(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    full_data: TransactionData,
) -> pd.DataFrame:
    """Match k-fold for transaction data."""
    if isinstance(full_data, RepeatTransactionData):
        return match_kfold_rtdata(train_df, pred_df)
    raise NotImplementedError


def match_kfold_rtdata(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Match k-fold for repeat transaction data."""
    trans_pair = list(train_df["trans_id1"] + "_" + train_df["trans_id2"])
    score_df = pred_df[~(pred_df["trans_id1"] + "_" + pred_df["trans_id2"]).isin(trans_pair)]
    return score_df


def match_kfold_heddata(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Match k-fold for hedonic transaction data."""
    # Choose every other one.
    x1 = pred_df.iloc[::2]  # even rows
    x2 = pred_df.iloc[1::2]  # odd rows
    # return pred_df[np.unique(zip(x1, x2))]
    return pred_df.loc[pd.MultiIndex.from_tuples(zip(x1, x2)).unique()]


def forecast_error(
    is_obj: BaseHousePriceIndex,
    trans_data: TransactionData,
    forecast_length: int = 1,
    **kwargs: Any,
) -> pd.DataFrame:
    """Calculate forecast error.

    Args:
        is_obj (BaseHousePriceIndex): Index object.
        trans_data (TransactionData): Transaction data.
        forecast_length (int, optional): Forecast length.
            Defaults to 1.

    Returns:
        pd.DataFrame: Forecast error.
    """
    # Check classes.
    if not isinstance(is_obj, (HedonicIndex, RepeatTransactionIndex)):
        raise ValueError("First argument must be an index instance.")
    if not isinstance(trans_data, (HedonicTransactionData, RepeatTransactionData)):
        raise ValueError("'trans_data' argument must be a transaction data instance.")

    if is_obj.hpis is None:
        raise ValueError

    # Set start and end.
    start = int(is_obj.hpis[0].value.index[-1] + 1)
    end = int(is_obj.hpis[-1].value.index[-1] + 1)
    time_range = list(range(start, end + 1))

    # Get data.
    fc_preddata = [
        trans_data.create_forecast_periods(time_cut, forecast_length=forecast_length, train=False)
        for time_cut in time_range
    ]

    # Predict value.
    if "smooth" in kwargs and kwargs["smooth"] and (len(is_obj.hpis[0].smooth) > 0):
        index_name = "smooth"
    else:
        if "smooth" in kwargs and kwargs["smooth"]:
            raise ValueError("No smoothed indices found. Create them try again.")
        index_name = "value"

    fit_kwargs = {"error": "add", "trend": None, "seasonal": None}
    fc_data = [
        (x.smooth.values if index_name == "smooth" else x.value.values) for x in is_obj.hpis
    ]
    fc_forecasts = [
        np.r_[x, ETSModel(x, **fit_kwargs).fit(disp=False).forecast(steps=forecast_length)]
        for x in fc_data
    ]
    if isinstance(trans_data, TransactionData):
        df = trans_data.trans_df
        index_type = is_obj.model
    else:
        raise ValueError

    if df is None:
        raise ValueError

    # Iterate through forecasts and calculate errors.
    fc_error = []
    for preddata, forecasts in zip(fc_preddata, fc_forecasts):
        error = insample_error(
            pred_df=df.iloc[preddata],
            index_type=index_type,
            index=pd.Series(forecasts, index=np.arange(1, len(forecasts) + 1)),
        )
        fc_error.append(error)

    error_df = pd.concat(fc_error)

    # self.error_df = error_df
    # self.test_method = 'forecast'

    # If returning forecasts.
    # if return_forecasts:
    #    self.forecasts = fc_forecasts

    return error_df


def revision(
    series_obj: BaseHousePriceIndex,
    in_place: bool = False,
    smooth: bool = False,
) -> Union[BaseHousePriceIndex, pd.DataFrame]:
    """Calculate the revision of a series.

    This is done by calculating the difference between consecutive indexes
    (n to n+1) and then calculating the mean and median of the differences.
    This is done for each period and the results are returned as a DataFrame.

    Args:
        series_obj (BaseHousePriceIndex): Series object.
        in_place (bool, optional): If True, the revision is placed into the
            series object as a named list.
            Defaults to False.
        smooth (bool, optional): Smooth the index. If True, the revision is
            calculated based on the smoothed indices.
            Defaults to False.

    Returns:
         Revision and associated statistics as object or DataFrame.
    """
    # Check class.
    if not isinstance(series_obj, BaseHousePriceIndex):
        raise ValueError("'series_obj' must be an index instance.")

    if smooth and len(series_obj.hpis[0].smooth) > 0:
        # index_name = "smooth"
        indices = [hpi_obj.smooth for hpi_obj in series_obj.hpis]
    else:
        if smooth:
            raise Exception(
                "No smoothed indices found. Create them with 'smooth_series' and try again."
            )
        # index_name = "value"
        indices = [hpi_obj.value for hpi_obj in series_obj.hpis]

    # Calculate the differences in the indices (n to n+1).
    index_diffs = [(indices[i][:-1] - indices[i - 1]) for i in range(1, len(indices))]

    # Extract differences into lists by period (essentially transposing list).
    period_diffs = [list(x) for x in zip(*index_diffs)]

    # Convert to vector format in correct order.
    period_diffs = [list(reversed(x)) for x in period_diffs]

    # Calculate the mean and medians.
    period_means = [np.mean(x) for x in period_diffs]
    period_medians = [np.median(x) for x in period_diffs]

    # Package and return.
    rev_obj = pd.DataFrame(
        {
            "period": range(1, len(period_means) + 1),
            "mean": period_means,
            "median": period_medians,
        }
    )

    if in_place:
        if smooth:
            series_obj.revision_smooth = rev_obj
        else:
            series_obj.revision = rev_obj
        return series_obj

    return rev_obj


def volatility(
    index: Union[BaseHousePriceIndex, pd.DataFrame],
    window: int = 3,
    in_place: bool = False,
    smooth: bool = False,
) -> Union[BaseHousePriceIndex, pd.DataFrame]:
    """Calculate index volatility.

    Args:
        index (Union[BaseHousePriceIndex, pd.DataFrame]): Index object.
        window (int, optional): Window for calculations.
            Defaults to 3.
        in_place (bool, optional): Return index in place.
            Defaults to False.
        smooth (bool, optional): Smooth the index.
            Defaults to False.

    Returns:
        Index object with volatility calculation or DataFrame.
    """
    # Save index_obj for future returning.
    index_obj = index

    # Strip from HPI or HPIIndex objects.
    if isinstance(index_obj, (HedonicIndex, RepeatTransactionIndex)):
        if not smooth:
            df = index_obj.value
        else:
            df = index_obj.smooth
            # Check to make sure a NULL wasn't given by smooth.
            if df is None:
                raise ValueError("No smoothed index present. Please set 'smooth = False'.")
    elif isinstance(index_obj, (BaseHousePriceIndex)):
        if not smooth:
            df = index_obj.value
        else:
            df = index_obj.smooth
            # Check to make sure a NULL wasn't given by smooth.
            if df is None:
                raise ValueError("No smoothed index present. Please set 'smooth = False'.")
    else:
        df = index_obj

    if df is None:
        raise ValueError

    # Check window.
    if (
        isinstance(window, (int, float))
        and not np.isnan(window)
        and window > 0
        and window <= len(df) / 2
    ):
        window = int(round(window))
    else:
        raise ValueError(
            "'window' argument must be a positive integer less than half the length of the index."
        )

    # Calculate changes.
    deltas = pd.Series((df[1:].to_numpy() - df[:-1].to_numpy()) / df[:-1].to_numpy())

    # Calculate mean rolling std.
    iv = deltas.rolling(window, center=True).std()

    # Create object.
    vol_obj = pd.DataFrame(
        {
            "roll": iv,
            "mean": iv.mean(),
            "median": iv.median(),
        },
        index=df.index[iv.first_valid_index() : iv.last_valid_index() + 1],
    )

    # self.vol_obj = vol_obj
    # self.orig = index
    # self.window = window

    # If returing in place.
    if in_place:
        if smooth:
            index_obj.volatility_smooth = vol_obj
        else:
            index_obj.volatility = vol_obj
        return index_obj

    # If just returning result of volatility calculation.
    return vol_obj


def series_volatility(
    series_obj: BaseHousePriceIndex,
    window: int = 3,
    smooth: bool = False,
) -> BaseHousePriceIndex:
    """Calculate volatility for a series of indices.

    Args:
        series_obj (BaseHousePriceIndex): Object with index series.
        window (int, optional): Window for calculations.
            Defaults to 3.
        smooth (bool, optional): Smooth the index.
            Defaults to False.

    Returns:
        Index object with series volatility calculation.
    """
    for hpi in series_obj.hpis:
        hpi = volatility(hpi, window, smooth=smooth, in_place=True)
    return series_obj
