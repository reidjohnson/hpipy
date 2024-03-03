import numpy as np
import pandas as pd
import pytest

from hpipy.period_table import PeriodTable


@pytest.mark.usefixtures("toy_dataset")
def test_period_table(toy_dataset: pd.DataFrame) -> None:
    toy_dataset["trans_date"] = toy_dataset["sale_date"]

    pt_df = PeriodTable(toy_dataset).create_period_table("sale_date", periodicity="annual")
    assert len(pt_df.period_table) == 7
    assert len(pt_df.period_table.columns) == 4
    assert pt_df.period_table["name"].iloc[3] == "2013"

    pt_df = PeriodTable(toy_dataset).create_period_table("sale_date", periodicity="quarterly")
    assert len(pt_df.period_table) == 28
    assert len(pt_df.period_table.columns) == 4
    assert pt_df.period_table["name"].iloc[3] == "2010-4"

    pt_df = PeriodTable(toy_dataset).create_period_table("sale_date", periodicity="monthly")
    assert len(pt_df.period_table) == 84
    assert len(pt_df.period_table.columns) == 4
    assert pt_df.period_table["name"].iloc[3] == "2010-Apr"

    pt_df = PeriodTable(toy_dataset).create_period_table("sale_date", periodicity="weekly")
    assert len(pt_df.period_table) == 364
    assert len(pt_df.period_table.columns) == 4
    assert pt_df.period_table["name"].iloc[5] == "week: 2010-02-07 to 2010-02-13"

    pt_df = PeriodTable(toy_dataset).create_period_table("sale_date", periodicity="equalfreq")
    assert len(pt_df.period_table) == 84
    assert len(pt_df.period_table.columns) == 4
    assert pt_df.period_table["name"].iloc[83] == "equalfreq (30): 2016-10-29 to 2016-12-22"

    pt_df = PeriodTable(toy_dataset).create_period_table(
        "sale_date", periodicity="equalfreq", freq=15
    )
    assert len(pt_df.period_table) == 169
    assert len(pt_df.period_table.columns) == 4
    assert (pt_df.period_table["end_date"][168] - pt_df.period_table["start_date"][168]).days > 15

    pt_df = PeriodTable(toy_dataset).create_period_table(
        "sale_date", periodicity="equalfreq", freq=15, start="last"
    )
    assert len(pt_df.period_table) == 169
    assert len(pt_df.period_table.columns) == 4
    assert (pt_df.period_table["end_date"][0] - pt_df.period_table["start_date"][0]).days > 15

    pt_df = PeriodTable(toy_dataset).create_period_table(
        "sale_date", periodicity="equalfreq", freq=15, start="last", first_date="2010-01-01"
    )
    assert len(pt_df.period_table) == 169
    assert len(pt_df.period_table.columns) == 4
    assert pt_df.period_table["start_date"].iloc[0].strftime("%Y-%m-%d") == "2010-01-01"
    assert (pt_df.period_table["end_date"][0] - pt_df.period_table["start_date"][0]).days > 15

    pt_df = PeriodTable(toy_dataset).create_period_table(
        "sale_date", periodicity="equalfreq", freq=15, start="last", first_date="2010-01-11"
    )
    assert len(pt_df.period_table) == 169
    assert len(pt_df.period_table.columns) == 4
    assert (pt_df.period_table["end_date"][0] - pt_df.period_table["start_date"][0]).days > 15

    pt_df = PeriodTable(toy_dataset).create_period_table(
        "sale_date", periodicity="equalsample", nbr_periods=50
    )
    assert len(pt_df.period_table) == 50
    assert len(pt_df.period_table.columns) == 4
    assert pt_df.period_table["name"].iloc[49] == "period 50"
    assert (
        pt_df.period_table["end_date"].iloc[49] - pt_df.period_table["start_date"].iloc[49]
    ).days == 48

    toy_dataset_gap = toy_dataset.iloc[np.arange(1000, 5001, step=1000) - 1]
    pt = PeriodTable(toy_dataset_gap)
    assert len(pt.create_period_table("sale_date", periodicity="weekly").period_table) == 153
    assert len(pt.create_period_table("sale_date", periodicity="monthly").period_table) == 36
    assert len(pt.create_period_table("sale_date", periodicity="quarterly").period_table) == 13
    assert len(pt.create_period_table("sale_date", periodicity="annual").period_table) == 4

    with pytest.raises(TypeError):
        PeriodTable(toy_dataset).create_period_table()  # type: ignore


@pytest.mark.usefixtures("toy_dataset")
def test_date_to_period(toy_dataset: pd.DataFrame) -> None:
    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date", periodicity="monthly"
    )

    assert "trans_period" in sales_df.trans_df.columns
    assert "trans_date" in sales_df.trans_df.columns
    assert len(sales_df.period_table) == 84
    assert len(sales_df.period_table.columns) == 4

    with pytest.raises(KeyError):
        PeriodTable(toy_dataset[["pinx"]]).create_period_table(date="sale_date")

    with pytest.raises(TypeError):
        PeriodTable(toy_dataset).create_period_table(date="sale_price")

    # Check `min_date`.
    for min_date in ["2010-01-03", "2010-01-01"]:
        sales_df = PeriodTable(toy_dataset).create_period_table(
            date="sale_date", periodicity="monthly", min_date=min_date
        )
        assert sales_df.min_date == min_date

    # Check `min_date` clip.
    for min_date in ["2010-01-03", "2010-01-01"]:
        sales_df = PeriodTable(toy_dataset).create_period_table(
            date="sale_date",
            periodicity="monthly",
            min_date=min_date,
            adj_type="clip",
        )
        assert sales_df.min_date == min_date

    # Check `min_date`.
    for max_date, expected_date in [("2016-12-29", "2016-12-29"), ("2010-12-27", "2016-12-22")]:
        sales_df = PeriodTable(toy_dataset).create_period_table(
            date="sale_date",
            periodicity="monthly",
            max_date=max_date,
        )
        assert sales_df.max_date == expected_date

    # Check `max_date` clip.
    for max_date in ["2016-12-21", "2016-12-29"]:
        sales_df = PeriodTable(toy_dataset).create_period_table(
            date="sale_date",
            periodicity="monthly",
            max_date=max_date,
            adj_type="clip",
        )
        assert sales_df.max_date == max_date

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        periodicity="annual",
    )
    assert sales_df.trans_df["trans_period"].iloc[0] == 2

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        min_date="2009-12-31",
        periodicity="annual",
    )
    assert sales_df.trans_df["trans_period"].iloc[0] == 2

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        min_date="2011-12-31",
        adj_type="clip",
        periodicity="annual",
    )
    assert sales_df.trans_df["trans_period"].iloc[0] == 3

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        periodicity="monthly",
    )
    assert sales_df.trans_df["trans_period"].max() == 84

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        min_date="2009-12-31",
        periodicity="monthly",
    )
    assert sales_df.trans_df["trans_period"].max() == 84

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        min_date="2011-12-31",
        adj_type="clip",
        periodicity="monthly",
    )
    assert sales_df.trans_df["trans_period"].max() == 60

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        periodicity="q",
    )
    assert sales_df.trans_df["trans_period"].max() == 28

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        min_date="2009-12-31",
        periodicity="q",
    )
    assert sales_df.trans_df["trans_period"].max() == 28

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        min_date="2011-12-31",
        adj_type="clip",
        periodicity="q",
    )
    assert sales_df.trans_df["trans_period"].max() == 20

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        min_date="2012-01-01",
        adj_type="clip",
        periodicity="q",
    )
    assert sales_df.trans_df["trans_period"].max() == 20

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        periodicity="w",
    )
    assert sales_df.trans_df["trans_period"].max() == 364

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        min_date="2009-12-31",
        periodicity="w",
    )
    assert sales_df.trans_df["trans_period"].max() == 364

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        min_date="2011-12-31",
        adj_type="clip",
        periodicity="Weekly",
    )
    assert sales_df.trans_df["trans_period"].max() == 260

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        min_date="2012-01-01",
        adj_type="clip",
        periodicity="Weekly",
    )
    assert sales_df.trans_df["trans_period"].max() == 260

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        periodicity="ef",
        freq=12,
    )
    assert sales_df.trans_df["trans_period"].max() == 212

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        periodicity="equalfreq",
        freq=12,
        start="last",
    )
    assert sales_df.trans_df["trans_period"].max() == 212

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        periodicity="equalfreq",
        freq=12,
        start="first",
        first_date="2009-08-01",
    )
    assert sales_df.trans_df["trans_period"].max() == 225

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        periodicity="equalfreq",
        freq=12,
        start="end",
        last_date="2019-08-01",
    )
    assert sales_df.trans_df["trans_period"].max() == 213

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        periodicity="es",
        nbr_periods=35,
    )
    assert sales_df.trans_df["trans_period"].max() == 35

    sales_df = PeriodTable(toy_dataset).create_period_table(
        date="sale_date",
        periodicity="es",
        nbr_periods=2,
    )
    assert sales_df.trans_df["trans_period"].max() == 2

    with pytest.raises(TypeError):
        PeriodTable(toy_dataset).create_period_table(date="sale_price")
