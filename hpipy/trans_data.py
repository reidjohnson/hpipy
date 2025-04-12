"""Transaction data. Used to construct house price models."""

import itertools
import logging
from abc import abstractmethod
from typing import Any, Optional, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import pandas as pd

from hpipy.period_table import PeriodTable


class TransactionData:
    """Transaction data class.

    Args:
        trans_data (Union[PeriodTable, pd.DataFrame]): Data from which to
            create transactions.
    """

    trans_data: Union[PeriodTable, pd.DataFrame]
    trans_df: pd.DataFrame
    period_table: pd.DataFrame

    def __init__(self, trans_data: Union[PeriodTable, pd.DataFrame]):
        """Initialize transaction data."""
        self.trans_data: Union[PeriodTable, pd.DataFrame] = trans_data

    @staticmethod
    def _check_fields(
        prop_id: Optional[str],
        trans_id: Optional[str],
        price: Optional[str],
    ) -> None:
        """Check necessary fields.

        Args:
            prop_id (Optional[str]): Property identifier.
            trans_id (Optional[str]): Transaction identifier.
            price (Optional[str]): Price column name.
        """
        if prop_id is None:
            raise ValueError("'prop_id' field not found.")

        if trans_id is None:
            raise ValueError("'trans_id' field not found.")

        if price is None:
            raise ValueError("'price' field not found.")

    def _create_period_table(
        self,
        date: Optional[str],
        periodicity: Optional[str] = None,
        **kwargs: Any,
    ) -> Self:
        """Create period table.

        Args:
            date (Optional[str]): Date column name.
            periodicity (Optional[str], optional): Period frequency.
                Defaults to None.
        """
        # Create the `trans_df` if not provided.
        if not isinstance(self.trans_data, PeriodTable):
            if date is None:
                raise ValueError(
                    "You must provide the name of a field with date of transaction (date=)."
                )
            if periodicity is None:
                logging.warning("No periodicity (periodicity=) provided, defaulting to yearly.")
                periodicity = "yearly"

            self.trans_data = PeriodTable(self.trans_data).create_period_table(
                date, periodicity, **kwargs
            )

        return self

    def _create_forecast_periods(
        self,
        time_period: str,
        time_cut: int,
        forecast_length: int = 1,
        train: bool = True,
    ) -> np.ndarray:
        """Create forecast periods.

        Args:
            time_period (str): Time period column name.
            time_cut (int): Time threshold.
            forecast_length (int, optional): Forecast length.
                Defaults to 1.
            train (bool, optional): Training Boolean.
                Defaults to True.

        Returns:
            Forecasts as a NumPy array.
        """
        if not isinstance(time_cut, int) or time_cut < 1:
            raise ValueError
        if not isinstance(forecast_length, int) or forecast_length < 0:
            raise ValueError
        if train:
            time_periods = np.where(self.trans_df[time_period] < time_cut)[0]
        else:
            time_seq = np.arange(time_cut, time_cut + forecast_length) + 1
            time_periods = np.where(self.trans_df[time_period].isin(time_seq))[0]
        return time_periods

    @abstractmethod
    def create_transactions(
        self,
        prop_id: Optional[str],
        trans_id: Optional[str],
        price: Optional[str],
        date: Optional[str] = None,
        periodicity: Optional[str] = None,
    ) -> Self:
        """Abstract transaction creation method."""
        raise NotImplementedError

    @abstractmethod
    def create_forecast_periods(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Abstract transaction forecast period method."""
        raise NotImplementedError


class RepeatTransactionData(TransactionData):
    """Repeat transaction data."""

    def create_transactions(
        self,
        prop_id: Optional[str],
        trans_id: Optional[str],
        price: Optional[str],
        date: Optional[str] = None,
        periodicity: Optional[str] = None,
        seq_only: bool = False,
        min_period_dist: Optional[int] = None,
        **kwargs: Any,
    ) -> Self:
        """Create repeat transaction data.

        Args:
            prop_id (Optional[str]): Property identifier.
            trans_id (Optional[str]): Transaction identifier.
            price (Optional[str]): Price column name.
            date (Optional[str], optional): Date column name.
                Defaults to None.
            periodicity (Optional[str], optional): Periodicity of the data.
                Defaults to None.
            seq_only (Optional[bool], optional): Sequential only.
                Defaults to False.
            min_period_dist (Optional[int], optional): Minimum period distance.
                Defaults to None.
        """
        self._check_fields(prop_id, trans_id, price)
        self._create_period_table(date, periodicity, **kwargs)

        trans_df: pd.DataFrame = self.trans_data.trans_df
        period_table: pd.DataFrame = self.trans_data.period_table

        # Prepare input data.
        trans_df = (
            trans_df.rename(columns={prop_id: "prop_id", trans_id: "trans_id", price: "price"})
            .loc[:, ["prop_id", "trans_id", "trans_period", "price"]]
            .sort_values(  # order by id, then time, then desc by price
                ["prop_id", "trans_period", "price"],
                ascending=[True, True, False],
            )
            .assign(
                **{
                    "trans_period": lambda x: x["trans_period"].astype(int),
                    "temp": lambda x: x["prop_id"] + "_" + x["trans_period"].astype(str),
                }
            )
            .drop_duplicates("temp")  # remove any properties that sold twice in same time period
            .drop(columns="temp")
        )

        # Count occurrences for each property.
        # Keep those with 2 or more sales.
        repeat_trans_df = (
            trans_df.groupby("prop_id")  # group by property
            .size()  # count number of sales
            .reset_index()
            .rename(columns={0: "count"})
            .query("count > 1")  # remove solo sales
        )

        if len(repeat_trans_df) == 0:
            raise ValueError("No repeat sales found.")

        # Split into 2 sales and greater than two sales per property.
        rt2 = repeat_trans_df.query("count == 2")
        rt3 = repeat_trans_df.query("count > 2")

        # Create repeat sales for properties with exactly 2 sales.
        if len(rt2) > 0:
            # Extract original sales and sort by id, then time.
            x_df = trans_df.query(f"prop_id in {list(rt2['prop_id'].unique())}").sort_values(
                ["prop_id", "trans_period"]
            )

            # Separate into first and second sale.
            id_1 = ~x_df["prop_id"].duplicated()
            id_2 = x_df["prop_id"].duplicated()

            # Create dataframe of repeat sales.
            d2 = pd.DataFrame(
                {
                    "prop_id": x_df[id_1]["prop_id"].reset_index(drop=True),
                    "period_1": x_df[id_1]["trans_period"].reset_index(drop=True),
                    "period_2": x_df[id_2]["trans_period"].reset_index(drop=True),
                    "price_1": x_df[id_1]["price"].reset_index(drop=True),
                    "price_2": x_df[id_2]["price"].reset_index(drop=True),
                    "trans_id1": x_df[id_1]["trans_id"].reset_index(drop=True),
                    "trans_id2": x_df[id_2]["trans_id"].reset_index(drop=True),
                }
            )
        else:
            d2 = pd.DataFrame()

        if len(rt3) > 0:
            x_df = trans_df.query(f"prop_id in {list(rt3['prop_id'].unique())}")

            # Create a dataframe of combinations of repeat sales.
            s = x_df.groupby("prop_id").apply(
                lambda x: list(itertools.combinations(x["trans_id"], 2)), include_groups=False
            )
            d3 = pd.DataFrame(
                np.vstack([[[s.index[idx]] + list(x_i) for x_i in x] for idx, x in enumerate(s)]),
                columns=["prop_id", "trans_id1", "trans_id2"],
            )

            # Add time and price.
            d3 = (
                d3.merge(
                    x_df.loc[:, ["trans_id", "trans_period", "price"]],
                    left_on="trans_id1",
                    right_on="trans_id",
                    how="left",
                )
                .assign(period_1=lambda x: x["trans_period"])
                .assign(price_1=lambda x: x["price"])
                .drop(columns=["trans_id", "trans_period", "price"])
                .merge(
                    x_df.loc[:, ["trans_id", "trans_period", "price"]],
                    left_on="trans_id2",
                    right_on="trans_id",
                    how="left",
                )
                .assign(period_2=lambda x: x["trans_period"])
                .assign(price_2=lambda x: x["price"])
                .drop(columns=["trans_id", "trans_period", "price"])
            )
        else:
            d3 = pd.DataFrame()

        # Combine and order.
        repeat_trans_df = pd.concat([d2, d3], ignore_index=True)
        repeat_trans_df = repeat_trans_df.sort_values(["prop_id", "period_1", "period_2"])

        # Check for sequential only.
        if seq_only:
            repeat_trans_df = repeat_trans_df[~repeat_trans_df["trans_id1"].duplicated()]

        if min_period_dist is not None:
            repeat_trans_df = (
                repeat_trans_df.assign(pdist=lambda x: x["period_2"] - x["period_1"])
                .query("pdist >= @min_period_dist")
                .drop(columns="pdist")
                .reset_index(drop=True)
            )

        # Add unique ID.
        repeat_trans_df = repeat_trans_df.assign(pair_id=list(range(1, len(repeat_trans_df) + 1)))

        if repeat_trans_df is None or len(repeat_trans_df) == 0:
            raise ValueError("No repeat transactions created.")

        repeat_trans_df = repeat_trans_df.sort_values(["prop_id", "trans_id1"])

        self.trans_df = repeat_trans_df
        self.period_table = period_table

        return self

    def create_forecast_periods(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Create repeat transaction forecast periods.

        Returns:
            np.ndarray: Forecast periods as an array.
        """
        return self._create_forecast_periods("period_2", *args, **kwargs)


class HedonicTransactionData(TransactionData):
    """Hedonic transaction data."""

    def create_transactions(
        self,
        prop_id: Optional[str],
        trans_id: Optional[str],
        price: Optional[str],
        date: Optional[str] = None,
        periodicity: Optional[str] = None,
        **kwargs: Any,
    ) -> Self:
        """Create hedonic transaction data.

        Args:
            prop_id (Optional[str]): Property identifier.
            trans_id (Optional[str]): Transaction identifier.
            price (Optional[str]): Price column name.
            date (Optional[str], optional): Date column name.
                Defaults to None.
            periodicity (Optional[str], optional): Periodicity of the data.
                Defaults to None.
        """
        self._check_fields(prop_id, trans_id, price)
        self._create_period_table(date, periodicity, **kwargs)

        trans_df: pd.DataFrame = self.trans_data.trans_df
        period_table: pd.DataFrame = self.trans_data.period_table

        # Prepare input data.
        hedonic_trans_df = trans_df.rename(
            columns={prop_id: "prop_id", trans_id: "trans_id", price: "price"}
        ).sort_values(["prop_id", "trans_period", "price"], ascending=[True, True, False])

        # Remove any properties that sold twice in same time period.
        # hedonic_trans_df = hedonic_trans_df.drop_duplicates(["prop_id", "trans_period"])

        if hedonic_trans_df is None or len(hedonic_trans_df) == 0:
            logging.warning("No hedonic sales created.")
            return self

        hedonic_trans_df = hedonic_trans_df.sort_values(["trans_date", "prop_id", "trans_id"])

        self.trans_df = hedonic_trans_df
        self.period_table = period_table

        return self

    def create_forecast_periods(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Create hedonic transaction forecast periods.

        Returns:
            np.ndarray: Forecast periods as an array.
        """
        return self._create_forecast_periods("trans_period", *args, **kwargs)
