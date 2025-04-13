"""Time matrix. Used by repeat transaction models."""

import numpy as np
import pandas as pd


class TimeMatrixMixin:
    """Time matrix mixin class.

    The class provides a method to create a time matrix from a dataframe
    consisting of repeat transaction data. The method utilizes datetime data
    represented as periods and constructs a chronologically ordered matrix.
    """

    def create_time_matrix(self, repeat_trans_df: pd.DataFrame) -> pd.DataFrame:
        """Create a time matrix from a dataframe of repeat transactions.

        This function assumes the dataframe contains columns "period_1" and
        "period_2" that represent a pair of repeat transactions. The resulting
        dataframe consists of rows for each transaction pair and columns for
        each time period, with -1 indicating the first transaction in a pair,
        1 indicating the second transaction in a pair, and 0 otherwise.

        Args:
            repeat_trans_df (pd.DataFrame): Input DataFrame. Must contain
                "period_1" and "period_2" columns containing integer values
                representing periods (i.e., time series expressed as integer).

        Returns:
            pd.DataFrame: DataFrame with columns 'time_x', where each row
                represents a transaction pair and 'x' is a time period between
                the minimum and maximum periods in the input data.
        """
        # Extract start/end/diff.
        time_start = repeat_trans_df["period_1"].min()
        time_end = repeat_trans_df["period_2"].max()
        time_diff = time_end - time_start

        # Set up empty matrix.
        time_matrix = np.zeros((repeat_trans_df.shape[0], time_diff))

        # Fill in time matrix.
        for tm in range(time_start + 1, time_end + 1):
            time_matrix[repeat_trans_df["period_1"] == tm, tm - time_start - 1] = -1
            time_matrix[repeat_trans_df["period_2"] == tm, tm - time_start - 1] = 1

        # Create time matrix dataframe.
        time_matrix_df = pd.DataFrame(
            time_matrix, columns=[f"time_{x}" for x in range(time_start + 1, time_end + 1)]
        )

        return time_matrix_df
