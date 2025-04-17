"""Data feature and response transformer functions."""

import collections

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import pandas as pd


class FeatureTransformer:
    """Feature transformer to transform raw features based on feature type.

    It takes a dictionary containing the names of the features to be encoded
    as transformed input, along with an optional fill value for categorical
    values. The dictionary should have the keys 'nulls', 'numerics',
    'log_numerics', 'categoricals', and 'ordinals', each with a list of the
    corresponding feature names.

    The `fit` method can be used to calculate the median values and populate
    label encodings for the features, and the `transform` method can be used
    to add the transformed features to the dataframe.
    """

    def __init__(
        self,
        feature_dict: dict[str, list],
        fill_value: float | str = -float("inf"),
    ) -> None:
        """Initialize the feature transformer.

        Args:
            feature_dict (dict[str, list]): Feature dictionary.
            fill_value (float | str, optional): Missing fill value.
                Defaults to -float("inf").

        """
        self.feature_dict_ = feature_dict
        self.fill_value_ = fill_value
        self.num_medians_: dict[str, float] = {}
        self.cat_counts_: dict[str, dict] = {}
        self.cat_labels_: dict[str, dict[int | float | str, int]] = {}

        for dtype in ["nulls", "numerics", "log_numerics", "categoricals", "ordinals"]:
            if dtype not in self.feature_dict_:
                self.feature_dict_[dtype] = []

    def update_features(self, feature_dict: dict[str, list[str]]) -> None:
        """Update the feature dictionary attribute.

        Args:
            feature_dict (dict[str, list[str]]): Feature dictionary.

        """
        self.feature_dict_ = feature_dict

    def prepare_init_dict(
        self,
        X: dict[str, np.ndarray] | pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Prepare model initialization dictionary.

        Model initialization may require information on the input features,
        including the number of categories for categorical/ordinal features.
        Rather than passing the complete DataFrame for initialization, this
        function populates an initialization dictionary with a single row that
        has the necessary values.

        Args:
            X (dict[str, np.ndarray] | pd.DataFrame): Input data.

        Returns:
            dict[str, np.ndarray]: Feature names mapped to NumPy arrays.

        """
        # if isinstance(X, pd.DataFrame):
        #     X_dict = {str(k): np.array(v) for k, v in X.to_dict()}
        # else:
        #     X_dict = X
        X_dict = X

        # Use a single row of the data as an initializer.
        init_dict = {k: v[np.newaxis, 0] for k, v in X_dict.items()}

        # Populate each categorical/ordinal feature with its max encoding.
        for k in self.feature_dict_["categoricals"] + self.feature_dict_["ordinals"]:
            f_vals = list(self.cat_counts_[k].keys())
            f_vals = [v for v in f_vals if v != self.fill_value_]
            f_vals = [self.fill_value_, *sorted(set(f_vals))]
            k_max = max([self.cat_labels_[k][f] for f in f_vals])
            init_dict[k] = np.array([k_max])

        return init_dict

    def fit(self, X: pd.DataFrame, copy: bool = True) -> Self:
        """Fit the feature transformer.

        Args:
            X (pd.DataFrame): Input feature data.
            copy (bool, optional): Copy the input data.
                Defaults to True.

        """
        X = X.copy() if copy else X

        for f in self.feature_dict_["numerics"] + self.feature_dict_["log_numerics"]:
            if X[f].isnull().sum() > 0:
                X[f + "_isnull"] = 0
                X.loc[X[f].isnull(), f + "_isnull"] = 1
                self.feature_dict_["nulls"].append(f + "_isnull")

            if f in self.feature_dict_["log_numerics"]:
                X[f] = np.log1p(X[f])

            self.num_medians_[f] = X[f].median()

            X[f] = X[f].fillna(self.num_medians_[f])
            X[f] = X[f] - self.num_medians_[f]

        for f in self.feature_dict_["categoricals"] + self.feature_dict_["ordinals"]:
            if np.issubdtype(X[f].dtype, np.integer) or np.issubdtype(X[f].dtype, np.floating):
                X[f] = X[f].astype(np.float64)

            X[f] = X[f].fillna(self.fill_value_)
            X[f] = X[f].astype("category")
            X[f] = X[f].cat.as_ordered()

            if f not in self.cat_counts_:
                self.cat_counts_[f] = collections.defaultdict(int)
            for k, v in X[f].value_counts().items():
                self.cat_counts_[f][k] += v

            f_vals = [k for (k, v) in self.cat_counts_[f].items() if v >= 1]
            f_vals = [v for v in f_vals if v != self.fill_value_]
            f_vals = [self.fill_value_, *sorted(set(f_vals))]
            self.cat_labels_[f] = {v: i for i, v in enumerate(f_vals)}

        return self

    def transform(self, X: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """Transform the feature data.

        Args:
            X (pd.DataFrame): Input feature data.
            copy (bool, optional): Copy the input data.
                Defaults to True.

        Returns:
            pd.DataFrame: Transformed feature data.

        """
        X = X.copy() if copy else X

        for f in self.feature_dict_["nulls"]:
            X[f] = 0
            X.loc[X[f[:-7]].isnull(), f] = 1

        for f in self.feature_dict_["numerics"] + self.feature_dict_["log_numerics"]:
            if f in self.feature_dict_["log_numerics"]:
                X[f] = np.log1p(X[f])

            X[f] = X[f].fillna(self.num_medians_[f])
            X[f] = X[f] - self.num_medians_[f]

        for f in self.feature_dict_["categoricals"] + self.feature_dict_["ordinals"]:
            if np.issubdtype(X[f].dtype, np.integer) or np.issubdtype(X[f].dtype, np.floating):
                X[f] = X[f].astype(np.float64)

            X[f] = X[f].fillna(self.fill_value_)
            X[f] = X[f].astype("category")
            X[f] = X[f].cat.as_ordered()

            d = self.cat_labels_[f]
            if f in self.feature_dict_["ordinals"]:
                f_min_val, f_max_val = list(d.keys())[1], list(d.keys())[-1]
                f_min_idx = X[f].astype(float) < f_min_val
                f_max_idx = X[f].astype(float) > f_max_val
            X[f] = X[f].apply(lambda x, d=d: d[x] if x in d else d[self.fill_value_])
            if f in self.feature_dict_["ordinals"]:
                X[f] = X[f].astype("category").cat.set_categories(list(d.values()))
                X.loc[f_min_idx, f] = d[f_min_val]
                X.loc[f_max_idx, f] = d[f_max_val]

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the feature data.

        Args:
            X (pd.DataFrame): Input feature data.

        Returns:
            pd.DataFrame: Transformed feature data.

        """
        return self.fit(X).transform(X)


class ResponseTransformer:
    """Response transformer to transform raw continuous response values.

    It has two methods `fit` and `transform`, which can be used to fit the
    transformer to a set of response variables and transform those response
    variables, respectively. The `transform` method computes the median of the
    response variable, which it uses to center the response variable around 0.
    The `inverse_transform` method can be used to reverse the transformation
    applied by `transform`.
    """

    def __init__(self) -> None:
        """Initialize the response transformer."""
        self.median_ = np.nan

    def fit(self, y: pd.Series) -> Self:
        """Fit the response transformer.

        Args:
            y (pd.Series): Input response data.

        """
        self.median_ = y.median()
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        """Transform the response data.

        Args:
            y (pd.Series): Input response data.

        Returns:
             pd.Series: Transformed response data.

        """
        return np.log(y) - np.log(self.median_)

    def inverse_transform(self, y: pd.Series) -> pd.Series:
        """Inverse tansform the response data.

        Args:
            y (pd.Series): Input response data.

        Returns:
            pd.Series: Inverse transformed response data.

        """
        return np.exp(y + np.log(self.median_))

    def fit_transform(self, y: pd.Series) -> pd.Series:
        """Fit and transform the response data.

        Args:
            y (pd.Series): Input response data.

        Returns:
            pd.Series: Transformed response data.

        """
        return self.fit(y).transform(y)
