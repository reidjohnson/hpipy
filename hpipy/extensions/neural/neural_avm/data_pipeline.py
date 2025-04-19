"""Data pipeline functions."""

import copy
from collections.abc import Sequence

import pandas as pd

from .data_preprocessors import BaseFeaturePreprocessor
from .data_transformers import FeatureTransformer, ResponseTransformer


class DataPipeline:
    """Data pipeline to transform raw training and prediction data."""

    def __init__(
        self,
        feature_dict: dict[str, list[str]],
        response_col: str,
        feature_preprocessors: Sequence[BaseFeaturePreprocessor] | None = None,
        feature_transformer: FeatureTransformer | None = None,
        response_transformer: ResponseTransformer | None = None,
    ) -> None:
        """Initialize the data pipeline.

        Args:
            feature_dict (dict[str, list[str]]): Feature dictionary.
            response_col (str): Response column.
            feature_preprocessors (Sequence[BaseFeaturePreprocessor] | None,
                optional): Feature preprocessors.
                Defaults to None.
            feature_transformer (FeatureTransformer | None, optional):
                Feature transformer class.
                Defaults to None.
            response_transformer (ResponseTransformer | None, optional):
                Response transformer class.
                Defaults to None.

        """
        self.feature_dict_ = feature_dict
        self.response_col_ = response_col
        self.feature_preprocessors_ = feature_preprocessors
        self.feature_transformer_ = feature_transformer
        self.response_transformer_ = response_transformer

        self.feature_dict_trans: dict[str, list[str]] = {}

    @staticmethod
    def _get_feature_list(feature_dict: dict[str, list[str]]) -> list[str]:
        """Get flattened feature list.

        Args:
            feature_dict (dict[str, list[str]]): Feature dictionary.

        Returns:
            list[str]: Flattened feature list.

        """
        features = [f for type in feature_dict for f in feature_dict[type]]
        return sorted(set(features))

    def get_transformed_feature_dict(self) -> dict[str, list[str]]:
        """Get feature dictionary of transformed data.

        Returns:
            dict[str, list[str]]: Updated feature dictionary.

        """
        return self.feature_dict_trans

    def train_transform(self, df_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Transform and return training data.

        Args:
            df_train (pd.DataFrame): Input training data.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Transformed training X and y data.

        """
        df_train = df_train.copy(deep=True)
        feature_dict = copy.deepcopy(self.feature_dict_)

        if self.feature_preprocessors_ is not None:
            for preprocessor in self.feature_preprocessors_:
                feature_dict = preprocessor.update_features(feature_dict)

            for preprocessor in self.feature_preprocessors_:
                df_train = preprocessor.transform(df_train)

        features = self._get_feature_list(feature_dict)

        X_train = df_train[features].copy()
        y_train = df_train[self.response_col_]

        if self.feature_transformer_ is not None:
            self.feature_transformer_.update_features(feature_dict)
            X_train = self.feature_transformer_.fit_transform(X_train)

        self.feature_dict_trans = feature_dict

        X_train = X_train.reset_index(drop=True)

        if self.response_transformer_ is not None:
            y_train = self.response_transformer_.fit_transform(y_train)
        y_train = y_train.reset_index(drop=True)

        return X_train, y_train

    def predict_transform(
        self,
        df_predict: pd.DataFrame,
        max_train_date: str | None = None,
        override_date: bool = True,
        return_y: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series] | pd.DataFrame:
        """Transform and return prediction data.

        Args:
            df_predict (pd.DataFrame): Input prediction data.
            max_train_date (str | None, optional): Maximum training date.
                Defaults to None.
            override_date (bool, optional): Date override. If True, sets the
                prediction date for all rows to the maximum training date.
                Defaults to True.
            return_y (bool, optional): Return transformed true y values.
                Defaults to False.

        Returns:
            tuple[pd.DataFrame, pd.Series] | pd.DataFrame: Transformed
                prediction X (and y) data.

        """
        df_predict = df_predict.copy(deep=True)

        if len(df_predict) == 0:
            msg = "The DataFrame is empty."
            raise ValueError(msg)

        if override_date:
            if max_train_date is None:
                raise ValueError
            df_predict["sale_date"] = pd.to_datetime(max_train_date)

        if self.feature_preprocessors_ is not None:
            for preprocessor in self.feature_preprocessors_:
                df_predict = preprocessor.transform(df_predict)

        features = self._get_feature_list(self.feature_dict_trans)

        X_predict = df_predict[features].copy()

        if self.feature_transformer_ is not None:
            X_predict = self.feature_transformer_.transform(X_predict)

        if return_y:
            if self.response_col_ in df_predict:
                y_predict = df_predict[self.response_col_]
                if self.response_transformer_ is not None:
                    y_predict = self.response_transformer_.transform(y_predict)
            else:
                msg = f"Response column {self.response_col_} is missing."
                raise ValueError(msg)

        return (X_predict, y_predict) if return_y else X_predict
