"""Metrics utilities."""

import copy
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from hpipy.price_index import BaseHousePriceIndex, HedonicIndex, RepeatTransactionIndex
from hpipy.price_model import BaseHousePriceModel, RepeatTransactionModel
from hpipy.trans_data import HedonicTransactionData, RepeatTransactionData, TransactionData


def accuracy(
    hpi_obj: BaseHousePriceIndex,
    test_method: str = "insample",
    test_type: str | None = None,
    pred_df: TransactionData | pd.DataFrame | None = None,
    smooth: bool = False,
    in_place: bool = False,
    in_place_name: str = "accuracy",
    **kwargs: Any,
) -> BaseHousePriceIndex | pd.DataFrame:
    """Calculate the accuracy of an index.

    Args:
        hpi_obj (BaseHousePriceIndex): Index object.
        test_method (str, optional): Testing method.
            Defaults to "insample". Also supports "kfold".
        test_type (str, optional): Testing type.
            Defaults to None. If None, the test_type is inferred from the
            index object.
        pred_df (TransactionData | pd.DataFrame | None, optional): Prediction
            data.
            Defaults to None.
        smooth (bool, optional): Smooth the index. If True, the revision is
            calculated based on the smoothed indices.
            Defaults to False.
        in_place (bool, optional): Return accuracy in-place.
            Defaults to False.
        in_place_name (str, optional): Name of the attribute to store the
            accuracy in.
            Defaults to "accuracy".
        **kwargs: Additional keyword arguments.

    Returns:
        BaseHousePriceIndex | pd.DataFrame: Index object containing the
            accuracy or DataFrame.

    """
    # Check for class of hpi_obj.
    if not isinstance(hpi_obj, BaseHousePriceIndex):
        msg = "'hpi_obj' must be an index instance."
        raise ValueError(msg)

    # Check for allowed test_method.
    if test_method not in ["insample", "kfold"]:
        msg = "'test_method' must be one of 'insample' or 'kfold'."
        raise ValueError(msg)

    # Check for allowed test_method.
    if test_type is None:
        if isinstance(hpi_obj, RepeatTransactionIndex):
            test_type = "rt"
        else:
            test_type = "hed"

    elif test_type not in ["rt", "hed"]:
        msg = "'test_type' must be one of 'rt', 'hed'."
        raise ValueError(msg)

    # Check agreement between test_type and hpi_obj
    if isinstance(pred_df, pd.DataFrame):
        df: pd.DataFrame = pred_df
    else:
        df = hpi_obj.data.trans_df
    index_type = hpi_obj.model

    # Check for smooth.
    if smooth and not hasattr(hpi_obj, "smooth"):
        msg = "'hpi_obj' has no smoothed index. Please add one or set 'smooth' to False."
        raise ValueError(msg)

    # Clip pred_df to size of index.
    if test_type == "rt" and df["period_2"].max() > hpi_obj.periods.max():
        logging.info(
            f"Trimming prediction date down to period {hpi_obj.periods.max()} and before.",
        )
        df = df[df["period_2"] <= hpi_obj.periods.max()]

    # In-sample.
    if test_method == "insample":
        index = hpi_obj.value
        if smooth:
            index = hpi_obj.smooth
        accr_obj = _insample_error(pred_df=df, index_type=index_type, index=index)

    # K-fold.
    if test_method == "kfold":
        accr_obj = _kfold_error(hpi_obj=hpi_obj, pred_df=df, smooth=smooth, **kwargs)

    if in_place:
        setattr(hpi_obj, in_place_name, accr_obj)
        return hpi_obj

    return accr_obj


def _insample_error(
    pred_df: pd.DataFrame,
    index_type: BaseHousePriceModel,
    index: pd.Series,
) -> pd.DataFrame:
    """Calculate in-sample error.

    Args:
        pred_df (pd.DataFrame): Prediction data.
        index_type (BaseHousePriceModel): Index type.
        index (pd.Series): Index.

    Returns:
        pd.DataFrame: In-sample error.

    """
    if not isinstance(pred_df, pd.DataFrame):
        msg = "'pred_df' argument must be a dataframe."
        raise ValueError(msg)
    if isinstance(index_type, RepeatTransactionModel):
        return _insample_error_rtdata(pred_df, index)
    else:
        return _insample_error_heddata(pred_df, index)


def _insample_error_rtdata(pred_df: pd.DataFrame, index: pd.Series) -> pd.DataFrame:
    """Calculate in-sample error for repeat transaction index and data.

    Args:
        pred_df (pd.DataFrame): Prediction data.
        index (pd.Series): Index.

    Returns:
        pd.DataFrame: In-sample error.

    """
    # Calculate the index adjustment to apply.
    adj = []
    for idx2, idx1 in zip(pred_df["period_2"] - 1, pred_df["period_1"] - 1, strict=False):
        adj.append(index.iloc[int(idx2)] / index.iloc[int(idx1)])

    # Calculate a prediction price.
    pred_price = pred_df["price_1"] * adj

    # Measure the error (difference from actual).
    error = (pred_price - pred_df["price_2"]) / pred_df["price_2"]
    logerror = np.log(pred_price) - np.log(pred_df["price_2"])

    return pd.DataFrame(
        {
            "pair_id": pred_df["pair_id"],
            "rt_price": pred_df["price_2"],
            "pred_price": pred_price,
            "error": error,
            "log_error": logerror,
            "pred_period": pred_df["period_2"],
        },
    )


def _insample_error_heddata(
    pred_df: pd.DataFrame,
    index: pd.Series,
    **kwargs: Any,
) -> pd.DataFrame:
    """Calculate in-sample error for hedonic index and data.

    Args:
        pred_df (pd.DataFrame): Prediction data.
        index (pd.Series): Index.
        **kwargs: Additional keyword arguments.

    Returns:
        pd.DataFrame: In-sample error.

    """
    # Future method.
    raise NotImplementedError


def _kfold_error(
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
    if not isinstance(hpi_obj, HedonicIndex | RepeatTransactionIndex):
        msg = "hpi_obj argument must be of class 'hpi'."
        raise ValueError(msg)

    # Check pred_df.
    if not isinstance(pred_df, pd.DataFrame):
        msg = "'pred_df' argument must be a DataFrame."
        raise ValueError(msg)

    # Check k.
    if not isinstance(k, int | float) or k < 2:
        msg = "Number of folds ('k' argument) must be a positive integer greater than 1."
        raise ValueError(msg)

    # Check seed.
    if not isinstance(seed, int | float) or seed < 0:
        msg = "'seed' must be a non-negative integer."
        raise ValueError(msg)

    errors = []

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for _, test_idx in kf.split(hpi_obj.data.trans_df):
        train_df, score_df = _create_kfold_data(test_idx, hpi_obj.data, pred_df)

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
            if "smooth_order" in kwargs:
                smooth_order = kwargs["smooth_order"]
            index = k_index.smooth_index(order=smooth_order)

        # Calc errors.
        k_error = _insample_error(score_df, index_type=hpi_obj.model, index=index)
        errors.append(k_error)

    return pd.concat(errors)


def _create_kfold_data(
    score_ids: np.ndarray,
    full_data: TransactionData,
    pred_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create k-fold for transaction data.

    Args:
        score_ids (np.ndarray): Score IDs.
        full_data (TransactionData): Full data.
        pred_df (pd.DataFrame): Prediction data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and score data.

    """
    return _create_kfold_data_rtdata(score_ids, full_data, pred_df)


def _create_kfold_data_rtdata(
    score_ids: np.ndarray,
    full_data: TransactionData,
    pred_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create k-fold for repeat transaction data.

    Args:
        score_ids (np.ndarray): Score IDs.
        full_data (TransactionData): Full data.
        pred_df (pd.DataFrame): Prediction data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and score data.

    """
    train_df = full_data.trans_df.iloc[
        ~full_data.trans_df.reset_index(drop=True).index.isin(score_ids)
    ]
    score_df = _match_kfold(train_df, pred_df, full_data)
    return train_df, score_df


def _match_kfold(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    full_data: TransactionData,
) -> pd.DataFrame:
    """Match k-fold for transaction data.

    Args:
        train_df (pd.DataFrame): Train data.
        pred_df (pd.DataFrame): Prediction data.
        full_data (TransactionData): Full data.

    Returns:
        pd.DataFrame: Matched data.

    """
    if isinstance(full_data, RepeatTransactionData):
        return _match_kfold_rtdata(train_df, pred_df)
    else:
        return _match_kfold_heddata(train_df, pred_df)


def _match_kfold_rtdata(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Match k-fold for repeat transaction data.

    Args:
        train_df (pd.DataFrame): Train data.
        pred_df (pd.DataFrame): Prediction data.

    Returns:
        pd.DataFrame: Matched data.

    """
    trans_pair = list(train_df["trans_id1"] + "_" + train_df["trans_id2"])
    return pred_df[~(pred_df["trans_id1"] + "_" + pred_df["trans_id2"]).isin(trans_pair)]


def _match_kfold_heddata(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Match k-fold for hedonic transaction data.

    Args:
        train_df (pd.DataFrame): Train data.
        pred_df (pd.DataFrame): Prediction data.

    Returns:
        pd.DataFrame: Matched data.

    """
    # Choose every other one.
    mask1 = ~pred_df["trans_id1"].isin(train_df["trans_id"])
    x1 = pred_df[mask1].iloc[::2].index
    mask2 = ~pred_df["trans_id2"].isin(train_df["trans_id"])
    x2 = pred_df[mask2].iloc[::2].index

    selected_indices = x1.union(x2)
    return pred_df.loc[selected_indices]


def _forecast_error(
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
        **kwargs: Additional keyword arguments.

    Returns:
        pd.DataFrame: Forecast error.

    """
    # Check classes.
    if not isinstance(is_obj, HedonicIndex | RepeatTransactionIndex):
        msg = "First argument must be an index instance."
        raise ValueError(msg)
    if not isinstance(trans_data, HedonicTransactionData | RepeatTransactionData):
        msg = "'trans_data' argument must be a transaction data instance."
        raise ValueError(msg)

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
        if kwargs.get("smooth"):
            msg = "No smoothed indices found. Create them try again."
            raise ValueError(msg)
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
    for preddata, forecasts in zip(fc_preddata, fc_forecasts, strict=False):
        error = _insample_error(
            pred_df=df.iloc[preddata],
            index_type=index_type,
            index=pd.Series(forecasts, index=np.arange(1, len(forecasts) + 1)),
        )
        fc_error.append(error)

    return pd.concat(fc_error)

    # self.error_df = error_df
    # self.test_method = 'forecast'

    # If returning forecasts.
    # if return_forecasts:
    #    self.forecasts = fc_forecasts


def volatility(
    index: BaseHousePriceIndex | pd.DataFrame,
    window: int = 3,
    in_place: bool = False,
    smooth: bool = False,
) -> BaseHousePriceIndex | pd.DataFrame:
    """Calculate the volatility of an index.

    Args:
        index (BaseHousePriceIndex | pd.DataFrame): Index object.
        window (int, optional): Window for calculations.
            Defaults to 3.
        in_place (bool, optional): Return index in place.
            Defaults to False.
        smooth (bool, optional): Smooth the index.
            Defaults to False.

    Returns:
        BaseHousePriceIndex | pd.DataFrame: Index object with volatility
            calculation or DataFrame.

    """
    # Save index_obj for future returning.
    index_obj = index

    # Strip from HPI or HPIIndex objects.
    if isinstance(index_obj, HedonicIndex | RepeatTransactionIndex | BaseHousePriceIndex):
        if not smooth:
            df = index_obj.value
        else:
            df = index_obj.smooth
            # Check to make sure a NULL wasn't given by smooth.
            if df is None:
                msg = "No smoothed index present. Please set 'smooth = False'."
                raise ValueError(msg)
    else:
        df = index_obj

    if df is None:
        raise ValueError

    # Check window.
    if (
        isinstance(window, int | float)
        and not np.isnan(window)
        and window > 0
        and window <= len(df) / 2
    ):
        window = round(window)
    else:
        msg = (
            "'window' argument must be a positive integer less than half the length of the index."
        )
        raise ValueError(msg)

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


def revision(
    series_obj: BaseHousePriceIndex,
    in_place: bool = False,
    smooth: bool = False,
) -> BaseHousePriceIndex | pd.DataFrame:
    """Calculate the revision for a series of indices.

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
        BaseHousePriceIndex | pd.DataFrame: Revision and associated statistics
            as object or DataFrame.

    """
    # Check class.
    if not isinstance(series_obj, BaseHousePriceIndex):
        msg = "'series_obj' must be an index instance."
        raise ValueError(msg)

    if smooth and len(series_obj.hpis[0].smooth) > 0:
        # index_name = "smooth"
        indices = [hpi_obj.smooth for hpi_obj in series_obj.hpis]
    else:
        if smooth:
            msg = "No smoothed indices found. Create them with 'smooth_series' and try again."
            raise Exception(msg)
        # index_name = "value"
        indices = [hpi_obj.value for hpi_obj in series_obj.hpis]

    # Calculate the differences in the indices (n to n+1).
    index_diffs = [(indices[i][:-1] - indices[i - 1]) for i in range(1, len(indices))]

    # Extract differences into lists by period (essentially transposing list).
    period_diffs = [list(x) for x in zip(*index_diffs, strict=False)]

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
        },
    )

    if in_place:
        if smooth:
            series_obj.revision_smooth = rev_obj
        else:
            series_obj.revision = rev_obj
        return series_obj

    return rev_obj


def series_accuracy(
    series_obj: BaseHousePriceIndex,
    test_method: str = "insample",
    test_type: str = "rt",
    pred_df: TransactionData | pd.DataFrame | None = None,
    smooth: bool = False,
    summarize: bool = False,
    in_place: bool = False,
    in_place_name: str = "accuracy",
    **kwargs: Any,
) -> BaseHousePriceIndex | pd.DataFrame | list[pd.DataFrame]:
    """Calculate the accuracy for a series of indices.

    Args:
        series_obj (BaseHousePriceIndex): Series object.
        test_method (str, optional): Testing method.
            Defaults to "insample". Also supports "kfold" or "forecast".
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
        BaseHousePriceIndex | pd.DataFrame: Series object containing the
            accuracy or DataFrame.

    """
    # Check for allowed test_method.
    if test_method not in ["insample", "kfold", "forecast"]:
        msg = "'test_method' must be one of 'insample', 'kfold' or 'forecast'."
        raise ValueError(msg)

    # Check for allowed test_method.
    if test_type not in ["rt", "hed"]:
        msg = "'test_type' must be one of 'rt', 'hed'."
        raise ValueError(msg)

    # Check agreement between test_type and hpi_obj.
    if not (
        (isinstance(series_obj.data, RepeatTransactionData) and test_type == "rt")
        or (isinstance(series_obj.data, HedonicTransactionData) and test_type == "hed")
    ):
        if pred_df is None:
            msg = (
                f"When 'test_type' ({test_type}) does not match the 'hpi' object model type "
                f"({type(series_obj)}) you must provide an 'pred_df' object of the necessary "
                f"class, in this case: {test_type}."
            )
            raise ValueError(msg)
    else:
        trans_data: TransactionData = series_obj.data

    accr_dfs = []
    if test_method != "forecast":
        # Check for smooth indices.
        if smooth and not hasattr(series_obj.hpis[0], "smooth"):
            msg = "No smoothed indices found. Please add or set smooth to False."
            raise ValueError(msg)

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
            accr_df_i = accr_df_i.assign(index=idx).loc[:, ["index", *list(accr_df_i.columns)]]

            accr_dfs.append(accr_df_i)

        accr_df = (
            pd.concat(accr_dfs, ignore_index=True)
            .sort_values(by=["index", "pred_period"])
            .reset_index(drop=True)
        )

        # If summarizing.
        if summarize:
            accr_df = (
                accr_df.groupby("pair_id")
                .agg({"pred_price": "mean", "error": "mean", "log_error": "mean"})
                .reset_index()
            )

    # If it is a forecast method.
    else:
        accr_df = _forecast_error(
            is_obj=series_obj,
            trans_data=trans_data,
            smooth=smooth,
            **kwargs,
        )

    # Return if not in place.
    if not in_place:
        return accr_df

    # Add to series object.
    if smooth and in_place_name == "accuracy":
        in_place_name = "accuracy_smooth"
    setattr(series_obj, in_place_name, accr_df)

    return series_obj


def series_volatility(
    series_obj: BaseHousePriceIndex,
    window: int = 3,
    smooth: bool = False,
    in_place: bool = False,
    in_place_name: str = "volatility",
) -> BaseHousePriceIndex | pd.DataFrame:
    """Calculate volatility for a series of indices.

    Args:
        series_obj (BaseHousePriceIndex): Object with index series.
        window (int, optional): Window for calculations.
            Defaults to 3.
        smooth (bool, optional): Smooth the index.
            Defaults to False.
        in_place (bool, optional): Return index in place.
            Defaults to False.
        in_place_name (str, optional): Name of the in-place attribute.
            Defaults to "volatility".

    Returns:
        BaseHousePriceIndex | pd.DataFrame: Index object with series
            volatility calculation or DataFrame.

    """
    vol_dfs = []
    # Check for smooth indices.
    if smooth and not hasattr(series_obj.hpis[0], "smooth"):
        msg = "No smoothed indices found. Please add or set smooth to False."
        raise ValueError(msg)

    # Calculate volatility.
    for idx, hpi_obj in enumerate(series_obj.hpis):
        hpi_obj_i = copy.deepcopy(series_obj)
        hpi_obj_i.data = hpi_obj.data
        hpi_obj_i.value = hpi_obj.value
        if smooth:
            hpi_obj_i.smooth = hpi_obj.smooth

        vol_df_i = pd.DataFrame(volatility(hpi_obj_i, window, smooth=smooth, in_place=False))
        vol_df_i = vol_df_i.assign(index=idx, period=vol_df_i.index).loc[
            :, ["index", "period", *list(vol_df_i.columns)]
        ]

        vol_dfs.append(vol_df_i)

    vol_df = pd.concat(vol_dfs, ignore_index=True)

    # Return if not in place.
    if not in_place:
        return vol_df

    # Add to series object.
    if smooth and in_place_name == "volatility":
        in_place_name = "volatility_smooth"
    setattr(series_obj, in_place_name, vol_df)

    return series_obj
