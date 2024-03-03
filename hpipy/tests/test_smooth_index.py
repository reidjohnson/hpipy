import pandas as pd
import pytest

from hpipy.price_index import RepeatTransactionIndex
from hpipy.price_model import RepeatTransactionModel
from hpipy.trans_data import RepeatTransactionData


@pytest.mark.usefixtures("seattle_dataset")
def test_rt_create_trans(seattle_dataset: pd.DataFrame) -> None:
    repeat_trans_data = RepeatTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
    )

    model_base = RepeatTransactionModel(repeat_trans_data).fit(
        estimator="base",
        log_dep=True,
    )

    index_base = RepeatTransactionIndex.from_model(model_base)

    with pytest.raises(ValueError):
        index_base.smooth_index(order=-3)
