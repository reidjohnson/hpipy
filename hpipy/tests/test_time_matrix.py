import pandas as pd
import pytest

from hpipy.time_matrix import TimeMatrixMixin
from hpipy.trans_data import RepeatTransactionData


@pytest.mark.usefixtures("seattle_dataset")
def test_rt_create_trans(seattle_dataset: pd.DataFrame) -> None:
    """Test repeat transaction time matrix creation."""
    repeat_trans_data = RepeatTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
    )

    assert (
        len(TimeMatrixMixin().create_time_matrix(repeat_trans_data.trans_df.iloc[0:2000])) == 2000
    )
    assert (
        len(TimeMatrixMixin().create_time_matrix(repeat_trans_data.trans_df.iloc[0:2000]).columns)
        == len(repeat_trans_data.period_table) - 1
    )
