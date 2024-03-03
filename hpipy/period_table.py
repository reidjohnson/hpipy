"""Period table. Represents time periods for house price indices."""

import datetime
import logging
from typing import Any, Optional, Tuple, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import pandas as pd


def check_date(
    date: Optional[Union[str, datetime.date, datetime.datetime, pd.Series]],
    name: str,
) -> Union[None, datetime.datetime]:
    """Check and validate the provided date object.

    Checks that the date is either of type datetime or convertible to
    datetime. Provides error handling for incorrect date inputs.

    Parameters:
        date (Optional[Union[str, datetime.date, datetime.datetime,
            pd.Series]]): Date object that needs validation.
        name (str): Name for the 'date' argument being checked.

    Returns:
        Union[None, datetime.datetime]: Date converted to a datetime.datetime
            object. If the input date is None, the function returns None.

    Raises:
        ValueError: If date cannot be converted to a datetime object.
    """
    # If None, return None.
    if date is None:
        return None

    # If a number, give an error (stops from converting numbers to dates).
    # if not np.issubdtype(date.dtype, np.datetime64):
    #     raise TypeError("Date must be a datetime type.")

    # If any form of date/time, convert to datetime.
    if isinstance(date, (datetime.date, datetime.datetime)):
        date = pd.to_datetime(date)
    else:
        # Try to convert dates given as characters, such as '2000-01-01'.
        date = pd.to_datetime(date)
        if np.any(date is pd.NaT):
            raise ValueError(f"{name} argument must be a datetime type.")

    return date


class PeriodTable:
    """Period table class.

    Organizes transaction data into defined time periods.

    Args:
        trans_df (pd.DataFrame): Transaction data.
        period_table (Optional[pd.DataFrame]): Period table.

    Raises:
        TypeError: If 'trans_df' is not or does not inherit from a DataFrame.
    """

    trans_df: pd.DataFrame
    period_table: pd.DataFrame

    def __init__(
        self,
        trans_df: pd.DataFrame,
        period_table: Optional[pd.DataFrame] = None,
    ):
        """Initialize period table."""
        # Check that `trans_df` is a DataFrame.
        if not isinstance(trans_df, pd.DataFrame):
            raise TypeError("'trans_df' must be a DataFrame (or inherit from one).")

        self.trans_df = trans_df

        if period_table is not None:
            self.period_table = period_table

    @staticmethod
    def _check_date(
        trans_df: pd.DataFrame,
        date: str,
        min_date: Optional[str],
        max_date: Optional[str],
        adj_type: Optional[str],
    ) -> Tuple[pd.DataFrame, datetime.datetime, datetime.datetime]:
        """Check date fields and arguments."""
        if not np.issubdtype(trans_df[date].dtype, np.datetime64):
            raise TypeError("Date must be a datetime type.")

        # Check date fields.
        trans_df.loc[:, date] = check_date(trans_df[date], "date")
        min_datetime = check_date(min_date, "min_date")
        max_datetime = check_date(max_date, "max_date")

        # Set minimum date.
        if min_datetime is None:
            min_datetime = trans_df[date].min()
        else:
            if min_datetime > trans_df[date].min():
                if adj_type == "move":
                    logging.info(
                        "Supplied 'min_date' is greater than minimum of transactions. Adjusting."
                    )
                    min_datetime = trans_df[date].min()
                elif adj_type == "clip":
                    logging.info(
                        "Supplied 'min_date' date is greater than minimum of transactions. "
                        "Clipping transactions."
                    )
                    trans_df = trans_df[trans_df[date] >= min_datetime].copy()

        # Set maximum date.
        if max_datetime is None:
            max_datetime = trans_df[date].max()
        else:
            if max_datetime < trans_df[date].max():
                if adj_type == "move":
                    logging.info(
                        "Supplied 'max_date' is less than maximum of transactions. Adjusting."
                    )
                    max_datetime = trans_df[date].max()
                elif adj_type == "clip":
                    logging.info(
                        "Supplied 'max_date' is less than maximum of transactions. "
                        "Clipping transactions."
                    )
                    trans_df = trans_df[trans_df[date] <= max_datetime]

        # Set standardized date field.
        trans_df.loc[:, "trans_date"] = trans_df[date]

        return trans_df, min_datetime, max_datetime  # type: ignore

    @staticmethod
    def _check_periodicity(periodicity: Optional[str]) -> str:
        """Check periodicity argument."""
        # Check for periodicity.
        if periodicity is None:
            logging.warning("No 'periodicity' supplied, defaulting to 'annual'.")
            periodicity = "annual"

        periodicity_options = [
            "weekly",
            "monthly",
            "quarterly",
            "annual",
            "yearly",
            "w",
            "m",
            "q",
            "a",
            "y",
            "equalfreq",
            "ef",
            "equalsample",
            "es",
        ]
        periodicity = periodicity.lower()
        if periodicity not in periodicity_options:
            raise ValueError(
                "'periodicity' must be one of: 'weekly', 'monthly', 'quarterly', "
                "'annual', 'equalfreq' or 'equalsample'."
            )
        else:
            if periodicity in ["yearly", "y", "a"]:
                periodicity = "annual"
            elif periodicity == "q":
                periodicity = "quarterly"
            elif periodicity == "m":
                periodicity = "monthly"
            elif periodicity == "w":
                periodicity = "weekly"
            elif periodicity == "ef":
                periodicity = "equalfreq"
            elif periodicity == "es":
                periodicity = "equalsample"

        return periodicity

    @staticmethod
    def _create_periods(
        trans_df: pd.DataFrame,
        periodicity: str,
        nbr_periods: Optional[int] = None,
        freq: Optional[int] = None,
        start: Optional[str] = None,
        first_date: Optional[str] = None,
        last_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Create dataframe of periods of specified periodicity."""
        if periodicity == "annual":
            date_range = pd.date_range(
                trans_df["trans_date"].min().to_period("Y").to_timestamp(),
                trans_df["trans_date"].max().to_period("Y").to_timestamp(),
                freq="AS",
            )
            start_date = list(date_range)
            end_date = list(date_range.to_period("A").to_timestamp(how="end"))
            period = list(range(1, len(start_date) + 1))
            name = [f"{x.strftime('%Y')}" for x in start_date]
            data = {"start_date": start_date, "end_date": end_date, "period": period, "name": name}
        elif periodicity == "quarterly":
            date_range = pd.date_range(
                trans_df["trans_date"].min().to_period("Q").to_timestamp(),
                trans_df["trans_date"].max().to_period("Q").to_timestamp(),
                freq="QS",
            )
            start_date = list(date_range)
            end_date = list(date_range.to_period("Q").to_timestamp(how="end"))
            period = list(range(1, len(start_date) + 1))
            name = [
                # f"{x_start.strftime('%Y-%q')}"
                f"{x_start.to_period('Q').strftime('%Y-%q')}"  # bug workaround
                for x_start in start_date
            ]
            data = {"start_date": start_date, "end_date": end_date, "period": period, "name": name}
        elif periodicity == "monthly":
            date_range = pd.date_range(
                trans_df["trans_date"].min().to_period("M").to_timestamp(),
                trans_df["trans_date"].max().to_period("M").to_timestamp(),
                freq="MS",
            )
            start_date = list(date_range)
            end_date = list(date_range.to_period("M").to_timestamp(how="end"))
            period = list(range(1, len(start_date) + 1))
            name = [x.strftime("%Y-%b") for x in start_date]
            data = {"start_date": start_date, "end_date": end_date, "period": period, "name": name}
        elif periodicity == "weekly":
            date_range = pd.date_range(
                trans_df["trans_date"].min().to_period("W").to_timestamp(),
                trans_df["trans_date"].max().to_period("W").to_timestamp(),
                freq="W-MON",
            )
            start_date = date_range.map(lambda x: x - pd.DateOffset(days=1)).tolist()
            end_date = (
                date_range.to_series().apply(lambda dt: dt + pd.DateOffset(weekday=5)).tolist()
            )
            period = list(range(1, len(start_date) + 1))
            name = [
                f"week: {x_start.strftime('%Y-%m-%d')} to {x_end.strftime('%Y-%m-%d')}"
                for x_start, x_end in zip(start_date, end_date)
            ]
            data = {"start_date": start_date, "end_date": end_date, "period": period, "name": name}
        elif periodicity == "equalfreq":
            if first_date is None:
                first = trans_df["trans_date"].min()
            else:
                first = pd.to_datetime(first_date)
                if first > trans_df["trans_date"].min():
                    logging.info("'first_date' is within bounds of data.")
            if last_date is None:
                last = trans_df["trans_date"].max()
            else:
                last = pd.to_datetime(last_date)
                if last < trans_df["trans_date"].max():
                    logging.info("'last_date' is within bounds of data.")

            if freq is None:
                freq = 30
                logging.warning(
                    "You did not supply a frequency ('freq = '). "
                    "Running at the default of 30 days."
                )

            if start is None:
                start = "first"
                logging.warning(
                    "You did not specify when you wanted to start counting ('start = "
                    "['first' | 'last']). Defaulting to starting from the first sale."
                )

            if start == "last":
                date_range = pd.date_range(
                    last.to_period("D").to_timestamp(),
                    first.to_period("D").to_timestamp(),
                    freq=f"-{freq}D",
                )
                date_range = date_range[::-1]
            else:
                date_range = pd.date_range(
                    first.to_period("D").to_timestamp(),
                    last.to_period("D").to_timestamp(),
                    freq=f"{freq}D",
                )
            start_date = list(date_range)
            end_date = list(
                np.array(
                    (pd.Series(date_range) + datetime.timedelta(days=freq)).dt.to_pydatetime()
                )
            )

            if start == "last" and start_date[0] > min(first, trans_df["trans_date"].min()):
                start_date[0] = min(first, trans_df["trans_date"].min())
            if end_date[-1] > last:
                i = -1
                while end_date[i - 1] > last:
                    i -= 1
                end_date[i - 1] = last
                start_date = start_date[:i]
                end_date = end_date[:i]
            period = list(range(1, len(start_date) + 1))
            name = [
                f"equalfreq ({freq}): {x_start.strftime('%Y-%m-%d')} to "
                f"{x_end.strftime('%Y-%m-%d')}"
                for x_start, x_end in zip(start_date, end_date)
            ]
            data = {"start_date": start_date, "end_date": end_date, "period": period, "name": name}
        elif periodicity == "equalsample":
            if nbr_periods is None:
                raise ValueError
            # Set quantiles.
            period_qtls = [(x / nbr_periods) for x in range(0, nbr_periods + 1)]
            date_qtls = trans_df["trans_date"].astype(int).quantile(period_qtls[:-1]).values
            date_qtls[0] = date_qtls[0] - 1
            start_date = pd.to_datetime(
                pd.Series(
                    pd.to_datetime(date_qtls, origin="1970-01-01") + pd.to_timedelta(1, unit="s")
                ).dt.strftime("%Y-%m-%d")
            ).tolist()
            end_date = (
                pd.to_datetime(
                    pd.concat(
                        [
                            pd.Series(pd.to_datetime(date_qtls[1:], origin="1970-01-01")),
                            pd.Series(pd.to_datetime(trans_df["trans_date"].max())),
                        ],
                        ignore_index=True,
                    ).dt.strftime("%Y-%m-%d")
                )
                - datetime.timedelta(days=1)
            ).to_list()
            period = list(range(1, nbr_periods + 1))
            name = [f"period {x}" for x in list(range(1, nbr_periods + 1))]
            data = {"start_date": start_date, "end_date": end_date, "period": period, "name": name}

        periods_df = pd.DataFrame(data)

        return periods_df

    def create_period_table(
        self,
        date: str,
        periodicity: Optional[str] = None,
        nbr_periods: Optional[int] = None,
        freq: Optional[int] = None,
        start: Optional[str] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        adj_type: Optional[str] = "move",
        **kwargs: Any,
    ) -> Self:
        """Create a period table from a transaction dataframe.

        Args:
            date (str): Date field.
            periodicity (Optional[str], optional): Periodicity of the table.
                Defaults to None.
            nbr_periods (Optional[int], optional): Number of periods (only
                used if `periodicity` is "equalsample").
                Defaults to None.
            freq (Optional[int], optional): Frequency in days (only used if
                `periodicity` is "equalfreq").
                Defaults to None.
            start (Optional[str], optional): Starting position (only used if
                `periodicity` is "equalfreq"). One of "first" or "last".
                Defaults to None.
            min_date (Optional[str], optional): Minimum date.
                Defaults to None.
            max_date (Optional[str], optional): Maximum dare.
                Defaults to None.
            adj_type (ptional[str], optional): Adjustment type.
                Defaults to "move".
        """
        trans_df = self.trans_df

        trans_df, min_datetime, max_datetime = self._check_date(
            trans_df, date, min_date, max_date, adj_type
        )
        periodicity = self._check_periodicity(periodicity)

        # Make period_table.
        period_table = self._create_periods(
            trans_df,
            periodicity,
            nbr_periods=nbr_periods,
            freq=freq,
            start=start,
            **kwargs,
        )

        # Add to `trans_df`.
        trans_df.loc[:, "trans_period"] = (
            pd.cut(
                trans_df["trans_date"],
                pd.concat(
                    [
                        period_table["start_date"],
                        period_table.loc[len(period_table) - 1 :, "end_date"]
                        + datetime.timedelta(days=1),
                    ],
                    ignore_index=True,
                ),
                right=False,
                labels=False,
            )
            + 1
        )

        # Check for missing periods.
        num_periods = trans_df["trans_period"].nunique()
        if num_periods < len(period_table):
            logging.info(
                f"Your choice of periodicity resulted in {len(period_table) - num_periods} "
                f"empty periods out of {len(period_table)} total periods."
            )
            if (len(period_table) - num_periods) / len(period_table) > 0.3:
                logging.warning(
                    "You may wish to set a coarser periodicity "
                    "or set different start and end dates."
                )

        self.trans_df = trans_df
        self.period_table = period_table

        self.min_date = min_datetime.strftime("%Y-%m-%d")
        self.max_date = max_datetime.strftime("%Y-%m-%d")

        return self
