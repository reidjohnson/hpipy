"""Data preprocessing functions."""

import abc
import copy
import datetime

import h3
import numpy as np
import pandas as pd


class BaseFeaturePreprocessor(abc.ABC):
    """Abstract feature preprocessor class."""

    input_cols: dict[str, str]
    output_cols: list[dict[str, str]]

    def __post_init__(self) -> None:
        """Post initialization."""
        self._has_required_attributes()

    def _has_required_attributes(self) -> None:
        """Check required class attributes."""
        req_attrs: list[str] = ["input_cols", "output_cols"]
        for attr in req_attrs:
            if not hasattr(self, attr):
                msg = f"Missing attribute: '{attr}'"
                raise AttributeError(msg)

    def update_features(self, feature_dict: dict[str, list[str]]) -> dict[str, list[str]]:
        """Update feature dictionary.

        Args:
            feature_dict (dict[str, list[str]]): Feature dictionary.

        Returns:
            dict[str, list[str]]: Updated feature dictionary.

        """
        feature_dict = copy.deepcopy(feature_dict)
        hpi_col = False
        for in_col in self.input_cols.values():
            for type in feature_dict:
                if in_col in feature_dict[type]:
                    if type == "hpi":
                        hpi_col = True
                    feature_dict[type].remove(in_col)
        for out_col in self.output_cols:
            if out_col["type"] != "hpi" or hpi_col:
                feature_dict[out_col["type"]].append(out_col["name"])
        return feature_dict

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract feature transform function."""
        raise NotImplementedError


class GeospatialPreprocessor(BaseFeaturePreprocessor):
    """Geospatial feature preprocessor.

    Transforms raw latitude and longitude inputs into spatial cell features.
    """

    def __init__(
        self,
        resolutions: int | list[int],
        latitude_col: str = "latitude",
        longitude_col: str = "longitude",
    ) -> None:
        """Initialize the feature preprocessor.

        Args:
            resolutions (int | list[int]): Geospatial (H3) cell resolutions.
            latitude_col (str, optional): Name of latitude column.
                Defaults to "latitude".
            longitude_col (str, optional): Name of longitude column.
                Defaults to "longitude".

        """
        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self.input_cols = {
            "latitude": latitude_col,
            "longitude": longitude_col,
        }

        self.output_cols: list[dict[str, str]] = []
        for res in resolutions:
            self.output_cols.append({"name": f"lat_lng_h3_{res}", "type": "categoricals"})

        self.resolutions_ = resolutions
        self.latitude_col = latitude_col
        self.longitude_col = longitude_col

        super().__post_init__()

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geospatial features.

        Args:
            df (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data.

        """
        for res in self.resolutions_:
            name = f"lat_lng_h3_{res}"
            mask = df[self.latitude_col].isnull() | df[self.longitude_col].isnull()
            df.loc[~mask, name] = df[~mask].apply(  # get H3 cells from latitude and longitude
                lambda x, res=res: h3.latlng_to_cell(
                    x[self.latitude_col],
                    x[self.longitude_col],
                    res,
                ),
                axis=1,
            )
            df.loc[mask, name] = np.nan
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform feature data.

        Args:
            df (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data.

        """
        return self._create_features(df)


class TemporalPreprocessor(BaseFeaturePreprocessor):
    """Temporal feature preprocessor.

    Transforms raw sale date input into trend and seasonality features.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        saledate_col: str = "sale_date",
    ) -> None:
        """Initialize the feature preprocessor.

        Args:
            start_date (str): Training start date.
            end_date (str): Training end date.
            saledate_col (str, optional): Name of sale date column.
                Defaults to "sale_date".

        """
        self.input_cols = {
            "saledate": saledate_col,
        }

        self.output_cols = [
            {"name": "weekssincestartdate", "type": "ordinals"},
            {"name": "weekofyearsin", "type": "numerics"},
            {"name": "weekofyearcos", "type": "numerics"},
        ]

        self.output_cols.extend(
            [
                {"name": "weekssincestartdate", "type": "hpi"},
                {"name": "weekofyearsin", "type": "hpi"},
                {"name": "weekofyearcos", "type": "hpi"},
            ],
        )

        self.start_date_ = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date_ = datetime.datetime.strptime(end_date, "%Y-%m-%d")

        self.total_days_ = (self.end_date_ - self.start_date_).days
        self.total_weeks_ = self.total_days_ / (365.25 / 52)

        super().__post_init__()

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features."""
        sale_daysago = (  # create periods working backward from the end date
            self.end_date_ - pd.to_datetime(df["sale_date"], errors="coerce")
        ).dt.days.values
        sale_weeksago = np.floor(np.clip(sale_daysago, 0, None) / (365.25 / 52))

        # Create trend and seasonality features with aligned time periods.
        df["weekssincestartdate"] = np.clip(np.floor(self.total_weeks_) - sale_weeksago, 1, None)
        df["weekofyearsin"] = np.sin(sale_weeksago * (2 * np.pi / 52))
        df["weekofyearcos"] = np.cos(sale_weeksago * (2 * np.pi / 52))

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform feature data.

        Args:
            df (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data.

        """
        return self._create_features(df)
