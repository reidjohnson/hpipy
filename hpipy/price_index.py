"""House price indices."""

import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from hpipy.period_table import PeriodTable
from hpipy.price_model import BaseHousePriceModel, HedonicModel, RepeatTransactionModel
from hpipy.trans_data import HedonicTransactionData, RepeatTransactionData, TransactionData
from hpipy.utils.stineman_interpolation import interpolate_stineman


class BaseHousePriceIndex(ABC):
    """Abstract base house price index class."""

    data: TransactionData
    model: BaseHousePriceModel
    name: pd.Series
    periods: pd.Series
    value: pd.Series
    index: Any
    imputed: np.ndarray
    smooth: Any
    volatility: pd.DataFrame
    volatility_smooth: pd.DataFrame
    revision: pd.DataFrame
    revision_smooth: pd.DataFrame

    def __init__(self, **kwargs: Any):
        """Initialize base house price index."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def coef_to_index(
        coef_df: pd.DataFrame,
        log_dep: bool,
        base_price: Union[int, float] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert coefficients to an index.

        Args:
            coef_df (pd.DataFrame): Coefficients.
            log_dep (bool): Log dependent variable.
            base_price (Union[int, float], optional): Base price.
                Defaults to 1.
        """
        coef_df.loc[coef_df["coefficient"].abs() < 1e-15, "coefficient"] = 0
        coef_df.loc[coef_df["coefficient"].eq(0) & (coef_df.index > 0), "coefficient"] = np.nan

        # Set up imputation identification vector.
        is_imputed = np.zeros(len(coef_df["coefficient"]))

        # Determine which index values need to be imputed.
        na_coef = coef_df["coefficient"].isnull()

        # If any, then work through imputation process.
        if na_coef[1:].sum() == len(na_coef[1:]):
            coef_df[:] = 0
        elif na_coef.sum() > 0:
            # Set all missing to imputed.
            is_imputed[na_coef] = 1

            min_valid_idx = np.where(~na_coef & coef_df["coefficient"] != 0)[0].min()

            # Fix cases where beginning is missing.
            if coef_df[na_coef].index.min() < min_valid_idx:
                logging.info("You are extrapolating beginning periods.")
                not_na = np.where((~na_coef) & (coef_df["coefficient"] != 0))[0]
                imp_to_0 = na_coef.index < np.min(not_na)
                coef_df.loc[imp_to_0, "coefficient"] = 0

            # Fix cases where end is missing.
            if len(coef_df["coefficient"]) in np.where(na_coef)[0]:
                logging.info("You are extrapolating ending periods.")
                not_na = np.where(~na_coef)[0]
                end_imp = np.arange(np.max(not_na), len(coef_df["coefficient"] + 1))
                end_coef = coef_df["coefficient"].ffill()
                coef_df.loc[end_imp.tolist(), "coefficient"] = end_coef[end_imp.tolist()]

            # coef_df["coefficient"] = coef_df["coefficient"].interpolate(method="linear")
            coef_df["coefficient"] = coef_df["coefficient"].transform(
                lambda x: pd.Series(interpolate_stineman(x.values))
            )

            n_imp = len(na_coef[na_coef])
            logging.info(f"Total of {n_imp} period{'s' if n_imp > 1 else ''} imputed.")

        max_period = int(coef_df["time"].max())

        # Convert estimate to an index value.
        if log_dep:  # coefficients represent log-ratio: log(Pt / P0)
            estimate = np.exp(coef_df["coefficient"])
            index_value = ((estimate) * 100)[:max_period]
        else:  # coefficients represent price delta: Pt - P0
            estimate = (coef_df["coefficient"] + base_price) / base_price
            index_value = ((estimate) * 100)[:max_period]
        index = index_value

        return index, is_imputed

    @classmethod
    def from_model(
        cls,
        model: BaseHousePriceModel,
        trans_data: Optional[TransactionData] = None,
        max_period: Optional[int] = None,
        smooth: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Create an index from a house price model.

        Args:
            model (BaseHousePriceModel): House price model object.
            trans_data (Optional[TransactionData], optional): Transaction data
                object.
                Defaults to None.
            max_period (Optional[int], optional): Maximum period for the
                index.
                Defaults to None.
            smooth (bool, optional): Smooth the index.
                Defaults to False.
        """
        if max_period is None:
            max_period = int(model.coefficients["time"].max())

        # Check max period.
        if not isinstance(max_period, (int, np.integer)):
            raise ValueError("'max_period' argument must be numeric/integer.")

        # Extract coefficients.
        coef_df = pd.DataFrame({"time": list(range(1, max_period + 1))})
        coef_df = coef_df.merge(model.coefficients, on="time", how="left")

        log_dep = False
        if "log_dep" in model.params and model.params["log_dep"]:
            log_dep = True

        coef_cols = ["time", "coefficient"]
        if len(coef_df.columns) > 2:
            partition_cols = [col for col in coef_df.columns if col not in coef_cols]
            unique_df = coef_df[partition_cols].drop_duplicates()
            index = np.empty(len(unique_df))
            is_imputed = np.empty(len(unique_df))
            for idx, row in unique_df.iterrows():
                coef_df_i = coef_df.merge(row.to_frame().T, on=partition_cols, how="inner")
                index_i, is_imputed_i = cls.coef_to_index(
                    coef_df_i, log_dep, base_price=model.base_price
                )
                index_i_df = pd.DataFrame({"index": index_i})
                for col in partition_cols:
                    index_i_df[col] = row[col]
                index[idx] = index_i_df
                is_imputed[idx] = is_imputed_i
        else:
            index, is_imputed = cls.coef_to_index(coef_df, log_dep, base_price=model.base_price)

        # Convert to a class object.
        instance = cls(
            model=model,
            name=model.periods["name"][:max_period],
            periods=model.periods["period"][:max_period],
            value=index,
            imputed=is_imputed[:max_period],
        )

        if not isinstance(instance, BaseHousePriceIndex):
            raise Exception("Converting model results to index failed.")

        if smooth:
            if "smooth_order" not in kwargs:
                smooth_order = 3
            else:
                smooth_order = kwargs["smooth_order"]

            instance = instance.smooth_index(order=smooth_order, in_place=True)
            if not hasattr(instance, "smooth"):
                raise Exception("Smoothing index failed.")

        if trans_data is not None:
            instance.data = trans_data

        assert instance.model is not None
        assert instance.name is not None
        assert instance.periods is not None
        assert instance.value is not None
        assert instance.imputed is not None

        return instance

    @staticmethod
    def get_data() -> type[TransactionData]:
        """Abstract transaction data fetcher method."""
        raise NotImplementedError

    @staticmethod
    def get_model() -> type[BaseHousePriceModel]:
        """Abstract model fetcher method."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _create_transactions(
        cls, trans_data: TransactionData, *args: Any, **kwargs: Any
    ) -> TransactionData:
        """Abstract transaction data creation method."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _create_model(cls, *args: Any, **kwargs: Any) -> BaseHousePriceModel:
        """Abstract model creation method."""
        raise NotImplementedError

    @classmethod
    def create_index(
        cls,
        trans_data: Union[TransactionData, PeriodTable, pd.DataFrame],
        prop_id: Optional[str] = None,
        trans_id: Optional[str] = None,
        price: Optional[str] = None,
        seq_only: bool = True,
        max_period: Optional[int] = None,
        smooth: bool = False,
        periodicity: Optional[str] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        adj_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Self:
        """Create the index.

        Args:
            trans_data (TransactionData): Input transaction data.
            prop_id (Optional[str], optional): Property identifier.
                Defaults to None.
            trans_id (Optional[str], optional): Transaction identifier.
                Defaults to None.
            price (Optional[str], optional): Price column name.
                Defaults to None.
            seq_only (bool, optional): Sequential only.
                Defaults to True.
            max_period (Optional[int], optional): Maximum index period.
                Defaults to None.
            smooth (bool, optional): Smooth the index.
                Defaults to False.
            periodicity (Optional[str], optional): Periodicity of the index.
                Defaults to None.
            min_date (Optional[str], optional): Minimum date for the index.
                Defaults to None.
            max_date (Optional[str], optional): Maximum date for the index.
                Defaults to None.
            adj_type (Optional[str], optional): Adjustment type.
                Defaults to None.
        """
        # Check if trans_data has a trans_df object.
        if not isinstance(trans_data, TransactionData):
            if not isinstance(trans_data, PeriodTable):
                if "date" not in kwargs:
                    raise ValueError(
                        "When supplying a raw dataframe to the 'trans_df' object, a valid 'date' "
                        "argument must be supplied."
                    )

                # Create period table object.
                trans_data = PeriodTable(trans_data).create_period_table(
                    date=kwargs["date"],
                    periodicity=periodicity,
                    min_date=min_date,
                    max_date=max_date,
                    adj_type=adj_type,
                )

            if trans_id is None:
                raise ValueError(
                    "When not supplying an 'hpidata' object, a 'trans_id' argument must be "
                    "supplied."
                )

            if prop_id is None:
                raise ValueError(
                    "When not supplying an 'hpidata' object, a 'prop_id' argument must be "
                    "supplied."
                )

            if price is None:
                raise ValueError(
                    "When not supplying an 'hpidata' object, a 'price' argument must be supplied."
                )

            # Create transaction data object.
            trans_data = cls._create_transactions(
                trans_data,
                prop_id=prop_id,
                trans_id=trans_id,
                price=price,
                date=kwargs["date"],
                periodicity=periodicity,
                seq_only=seq_only,
            )

        if not hasattr(trans_data, "trans_df"):
            raise ValueError("Converting sales data to sales object failed.")

        model = cls._create_model(trans_data, **kwargs)

        if not isinstance(model, BaseHousePriceModel):
            raise Exception("Estimating model failed.")

        # Convert to an index.
        index = cls.from_model(
            model,
            trans_data=trans_data,
            max_period=max_period,
            smooth=smooth,
            **kwargs,
        )

        return index

    def create_series(
        self,
        train_period: int = 12,
        max_period: Optional[int] = None,
        **kwargs: Any,
    ) -> Self:
        """Create a series from the index.

        Args:
            train_period (int, optional): Train period for the series.
                Defaults to 12.
            max_period (Optional[int], optional): Maximum period for the
                series.
                Defaults to None.
        """
        if self.model is None:
            raise ValueError
        if self.data is None:
            raise ValueError

        # Check training period.
        if not isinstance(train_period, (int, float)):
            raise ValueError("'train_period' must be a single numeric value.")

        train_period = int(train_period)

        # Check for alternate max period and its allowed value.
        if max_period is None or max_period > self.model.periods.period.max():
            max_period = self.model.periods.period.max()

        if max_period is None:
            raise ValueError

        # Ensure training period is no greater than specified max or index max.
        if train_period >= min(self.model.periods.period.max(), max_period):
            raise ValueError(
                "'train_period' is greater than the length of the "
                "index and/or the 'max_period' argument."
            )

        # Trim by time.
        time_range = list(range(train_period + 1, max_period + 1))
        time_range += [time_range[-1] + 1]  # ensures the proper number of periods are forecasted

        # Generate row ids for the training data.
        row_ids = [self.data.create_forecast_periods(time, train=True) for time in time_range]
        has_empty_period = any(len(row_id) == 0 for row_id in row_ids)
        if has_empty_period:
            logging.info("Some periods have no data. Removing them from series estimation.")
            time_range = [
                t for t in time_range if len(self.data.create_forecast_periods(t, train=True)) > 0
            ]
            row_ids = [row_id for row_id in row_ids if len(row_id) > 0]

        # Run models, indices and combine into HPI objects.
        is_hpis = []
        for idx in range(len(time_range)):
            data = copy.deepcopy(self.data)
            model = copy.deepcopy(self.model)
            if isinstance(self.data, TransactionData):
                data.trans_df = data.trans_df.iloc[row_ids[idx]]
            model_i = self._create_model(data, **model.params, **kwargs)
            index_i = copy.deepcopy(
                self.from_model(
                    model_i,
                    trans_data=data,
                    max_period=time_range[idx] - 1,
                )
            )
            is_hpis.append(index_i)

        self.hpis = is_hpis

        return self

    def smooth_index(
        self,
        order: Union[list[int], int] = 3,
        in_place: bool = False,
    ) -> Union[Self, pd.Series]:
        """Smooth the index.

        Args:
            order (Union[list[int], int], optional): Smoothing order.
                Defaults to 3.
            in_place (bool, optional): Smooth in place or return new object.
                Defaults to False.

        Returns:
            Self with smoothed index or new smoothed index object.
        """
        if not isinstance(order, list):
            order = [order]

        if self.value is None:
            raise ValueError

        # Check order.
        for idx, o_i in enumerate(order):
            if np.all(
                isinstance(o_i, (int, float))
                and not np.isnan(o_i)
                and o_i > 0
                and o_i <= len(self.value) / 2
            ):
                order[idx] = int(round(o_i, 0))
            else:
                raise ValueError(
                    "'order' argument must be a positive integer or list of positive integers "
                    " less than half the length of the index."
                )

        # Create smoothed index (retain existing).
        s_index = self.value

        # Smooth with moving average (multiple orders can be done in sequence).
        for o_i in order:
            s_index = s_index.rolling(window=o_i, center=True).mean()

        # Deal with NAs (should be NAs on the tail ends of the smoothing).
        na_smooth = np.where(s_index.isnull())[0]

        # Fill in low end NAs with original index.
        na_low = na_smooth[na_smooth < len(s_index) / 2]
        s_index.iloc[na_low] = self.value.iloc[na_low]

        # Fill in high-end NAs with forecasted values (off of smoothed).
        na_high = na_smooth[na_smooth >= len(s_index) / 2]
        high_fc = (
            ETSModel(s_index[: na_high[0]], error="add", trend=None, seasonal=None)
            .fit(disp=False)
            .forecast(steps=len(na_high))
        )

        new_high = (high_fc.mean() + self.value.iloc[na_high]) / 2
        s_index[na_high] = new_high
        sm_index = pd.Series(s_index, name="indexsmooth")

        self.order = order

        # If returning in place.
        if in_place:
            self.smooth = sm_index
            return self

        # If just returning result of smoothing.
        return sm_index

    def smooth_series(self, order: int = 3) -> Self:
        """Apply smoothing to all indices.

        Args:
            order (int) Smoothing order.
                Defaults to 3.
        """
        for hpi in self.hpis:
            hpi = hpi.smooth_index(order, in_place=True)
        return self


class RepeatTransactionIndex(BaseHousePriceIndex):
    """Repeat transaction house price index.

    This class implements the repeat transaction methodology for constructing
    house price indices. It uses pairs of transactions for the same property
    to estimate price changes over time, controlling for property-specific
    characteristics that remain constant between sales.

    Parameters
    ----------
    data : TransactionData
        The transaction data used to construct the index.
    model : BaseHousePriceModel
        The underlying price model used to estimate the index.
    name : pd.Series
        The names of the time periods in the index.
    periods : pd.Series
        The time periods covered by the index.
    value : pd.Series
        The index values.
    index : Any
        The index values in a different format.
    imputed : np.ndarray
        Boolean array indicating which periods were imputed.
    smooth : Any
        Smoothed version of the index values.
    volatility : pd.DataFrame
        Volatility measures for the index.
    volatility_smooth : pd.DataFrame
        Smoothed volatility measures.
    revision : pd.DataFrame
        Revision measures for the index.
    revision_smooth : pd.DataFrame
        Smoothed revision measures.

    Methods
    -------
    get_data()
        Returns the appropriate transaction data class.
    get_model()
        Returns the appropriate price model class.
    _create_transactions(trans_data, *args, **kwargs)
        Creates transaction data from input data.
    _create_model(trans_data, estimator, log_dep, **kwargs)
        Creates a price model from transaction data.
    create_index(trans_data, *args, **kwargs)
        Creates a new index from transaction data.
    create_series(train_period=12, max_period=None, **kwargs)
        Creates a series of indices for different time periods.
    smooth_index(order=3, in_place=False)
        Smooths the index values.
    smooth_series(order=3)
        Smooths a series of indices.

    Examples
    --------
    >>> import pandas as pd
    >>> from hpipy.price_index import RepeatTransactionIndex
    >>> # Create sample transaction data.
    >>> data = pd.DataFrame({
    ...     "property_id": [1, 1, 2, 2],
    ...     "transaction_id": [1, 2, 3, 4],
    ...     "price": [200000, 250000, 300000, 350000],
    ...     "date": ["2020-01", "2021-01", "2020-02", "2021-02"],
    ... })
    >>> # Create index.
    >>> index = RepeatTransactionIndex.create_index(
    ...     data,
    ...     prop_id="property_id",
    ...     trans_id="transaction_id",
    ...     price="price",
    ...     periodicity="M",
    ...     min_date="2020-01",
    ...     max_date="2021-02",
    ... )
    """

    @staticmethod
    def get_data() -> type[TransactionData]:
        """Get repeat transaction data."""
        return RepeatTransactionData

    @staticmethod
    def get_model() -> type[BaseHousePriceModel]:
        """Get repeat transaction model."""
        return RepeatTransactionModel

    @classmethod
    def _create_transactions(
        cls,
        trans_data: pd.DataFrame,
        *args: Any,
        **kwargs: Any,
    ) -> TransactionData:
        """Create repeat transaction data."""
        return cls.get_data()(trans_data).create_transactions(*args, **kwargs)

    @classmethod
    def _create_model(
        cls,
        trans_data: TransactionData,
        estimator: str,
        log_dep: bool,
        **kwargs: Any,
    ) -> BaseHousePriceModel:
        """Create a repeat transaction house price model."""
        return cls.get_model()(trans_data).fit(estimator=estimator, log_dep=log_dep, **kwargs)


class HedonicIndex(BaseHousePriceIndex):
    """Hedonic house price index.

    This class implements the hedonic methodology for constructing house price
    indices. It uses property characteristics to control for quality
    differences between properties and estimates the pure price change over
    time.

    Parameters
    ----------
    data : TransactionData
        The transaction data used to construct the index.
    model : BaseHousePriceModel
        The underlying price model used to estimate the index.
    name : pd.Series
        The names of the time periods in the index.
    periods : pd.Series
        The time periods covered by the index.
    value : pd.Series
        The index values.
    index : Any
        The index values in a different format.
    imputed : np.ndarray
        Boolean array indicating which periods were imputed.
    smooth : Any
        Smoothed version of the index values.
    volatility : pd.DataFrame
        Volatility measures for the index.
    volatility_smooth : pd.DataFrame
        Smoothed volatility measures.
    revision : pd.DataFrame
        Revision measures for the index.
    revision_smooth : pd.DataFrame
        Smoothed revision measures.

    Methods
    -------
    get_data()
        Returns the appropriate transaction data class.
    get_model()
        Returns the appropriate price model class.
    _create_transactions(trans_data, *args, **kwargs)
        Creates transaction data from input data.
    _create_model(trans_data, dep_var, ind_var, **kwargs)
        Creates a price model from transaction data.
    create_index(trans_data, *args, **kwargs)
        Creates a new index from transaction data.
    create_series(train_period=12, max_period=None, **kwargs)
        Creates a series of indices for different time periods.
    smooth_index(order=3, in_place=False)
        Smooths the index values.
    smooth_series(order=3)
        Smooths a series of indices.

    Examples
    --------
    >>> import pandas as pd
    >>> from hpipy.price_index import HedonicIndex
    >>> # Create sample transaction data.
    >>> data = pd.DataFrame({
    ...     "property_id": [1, 2, 3, 4],
    ...     "transaction_id": [1, 2, 3, 4],
    ...     "price": [200000, 250000, 300000, 350000],
    ...     "date": ["2020-01", "2020-01", "2021-01", "2021-01"],
    ...     "sqft": [1500, 1800, 2000, 2200],
    ...     "bedrooms": [3, 3, 4, 4],
    ... })
    >>> # Create index.
    >>> index = HedonicIndex.create_index(
    ...     data,
    ...     prop_id="property_id",
    ...     trans_id="transaction_id",
    ...     price="price",
    ...     periodicity="M",
    ...     min_date="2020-01",
    ...     max_date="2021-01",
    ...     ind_var=["sqft", "bedrooms"],
    ... )
    """

    @staticmethod
    def get_data() -> type[TransactionData]:
        """Get hedonic transaction data."""
        return HedonicTransactionData

    @staticmethod
    def get_model() -> type[BaseHousePriceModel]:
        """Get hedonic model."""
        return HedonicModel

    @classmethod
    def _create_transactions(
        cls,
        trans_data: pd.DataFrame,
        *args: Any,
        **kwargs: Any,
    ) -> TransactionData:
        """Create hedonic transaction data."""
        return cls.get_data()(trans_data).create_transactions(*args, **kwargs)

    @classmethod
    def _create_model(
        cls,
        trans_data: TransactionData,
        dep_var: str,
        ind_var: str,
        **kwargs: Any,
    ) -> BaseHousePriceModel:
        """Create a hedonic house price model."""
        if dep_var is not None and ind_var is not None:
            return cls.get_model()(trans_data).fit(dep_var=dep_var, ind_var=ind_var, **kwargs)
        raise ValueError(
            "A dependent (dep_var) and independent variables (ind_var) must be provided.",
        )
