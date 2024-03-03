import pandas as pd
import pytest

from hpipy.period_table import PeriodTable
from hpipy.trans_data import HedonicTransactionData, RepeatTransactionData


@pytest.mark.usefixtures("seattle_dataset")
def test_rt_create_trans(seattle_dataset: pd.DataFrame) -> None:
    sales_df = PeriodTable(seattle_dataset).create_period_table(
        date="sale_date",
        periodicity="monthly",
    )

    repeat_trans_data = RepeatTransactionData(sales_df).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
    )
    assert len(repeat_trans_data.trans_df) == 5102

    # Min date with move.
    repeat_trans_data = RepeatTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
        min_date="2012-03-21",
    )
    assert len(repeat_trans_data.trans_df) == 5102

    # Min date with adj.
    repeat_trans_data = RepeatTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
        min_date="2012-03-21",
        adj_type="clip",
    )
    assert len(repeat_trans_data.trans_df) == 2827

    # Max with move.
    repeat_trans_data = RepeatTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
        max_date="2015-03-21",
    )
    assert len(repeat_trans_data.trans_df) == 5102

    # Max with clip.
    repeat_trans_data = RepeatTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
        max_date="2014-03-21",
        adj_type="clip",
    )
    assert len(repeat_trans_data.trans_df) == 1148

    repeat_trans_data = RepeatTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
        seq_only=True,
    )
    assert len(repeat_trans_data.trans_df) == 4823

    repeat_trans_data = RepeatTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
        seq_only=True,
        min_period_dist=12,
    )
    assert len(repeat_trans_data.trans_df) == 3795

    with pytest.raises(ValueError):
        RepeatTransactionData(seattle_dataset).create_transactions(
            prop_id="pinx",
            trans_id="sale_id",
            price="sale_price",
            date="sale_date",
            periodicity="mocnthly",
        )

    with pytest.raises(TypeError):
        RepeatTransactionData(seattle_dataset).create_transactions(
            prop_id="pinx",
            trans_id="sale_id",
            price="sale_price",
            date="sale_price",
            periodicity="monthly",
        )

    with pytest.raises(TypeError):
        RepeatTransactionData(seattle_dataset).create_transactions(  # type: ignore
            date="sale_date",
            periodicity="monthly",
        )


@pytest.mark.usefixtures("seattle_dataset")
def test_hed_create_trans(seattle_dataset: pd.DataFrame) -> None:
    sales_df = PeriodTable(seattle_dataset).create_period_table(
        date="sale_date",
        periodicity="monthly",
    )

    hedonic_trans_data = HedonicTransactionData(sales_df).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
    )
    assert len(hedonic_trans_data.trans_df) == len(sales_df.trans_df)

    # Min date with move.
    hedonic_trans_data = HedonicTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
        min_date="2012-03-21",
    )
    assert len(hedonic_trans_data.trans_df) == len(sales_df.trans_df)

    # Min date with adj.
    hedonic_trans_data = HedonicTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
        min_date="2012-03-21",
        adj_type="clip",
    )
    assert len(hedonic_trans_data.trans_df) == 34097

    # Max with move.
    hedonic_trans_data = HedonicTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
        max_date="2015-03-21",
    )
    assert len(hedonic_trans_data.trans_df) == 43313

    # Max with clip.
    hedonic_trans_data = HedonicTransactionData(seattle_dataset).create_transactions(
        prop_id="pinx",
        trans_id="sale_id",
        price="sale_price",
        date="sale_date",
        periodicity="monthly",
        max_date="2014-03-21",
        adj_type="clip",
    )
    assert len(hedonic_trans_data.trans_df) == 21669
