"""House price models. Used to construct house price indices."""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import OneHotEncoder

from hpipy.time_matrix import TimeMatrixMixin
from hpipy.trans_data import TransactionData


class BaseHousePriceModel(ABC):
    """Abstract house price model class.

    Args:
        hpi_data (TransactionData): Input transaction data.
        **kwargs: Additional keyword arguments.

    Attributes:
        coefficients (pd.DataFrame): Coefficient data.
        model_obj (Any): Model object.
        periods (pd.DataFrame): Period data.
        base_price (float): Base price.
        params (dict[str, Any]): Parameters.

    """

    coefficients: pd.DataFrame
    model_obj: Any
    periods: pd.DataFrame
    base_price: float
    params: dict[str, Any]

    def __init__(self, hpi_data: TransactionData, **kwargs: Any) -> None:
        """Initialize base house price model."""
        if hpi_data.period_table is None:
            raise ValueError

        self.hpi_df: pd.DataFrame = hpi_data.trans_df
        self.period_table: pd.DataFrame = hpi_data.period_table

        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def _create_model(self, *args: Any, **kwargs: Any) -> Any:
        """Abstract model creation method."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Self:
        """Abstract model fitting method."""
        raise NotImplementedError


class RepeatTransactionModel(BaseHousePriceModel, TimeMatrixMixin):
    """Repeat transaction house price model."""

    def _create_model(
        self,
        repeat_trans_df: pd.DataFrame,
        period_table: pd.DataFrame,
        time_matrix: pd.DataFrame,
        price_diff: pd.Series,
        estimator: str,
        **kwargs: Any,
    ) -> LinearRegression:
        """Create a repeat transaction house price model.

        Args:
            repeat_trans_df (pd.DataFrame): The repeat transaction data.
            period_table (pd.DataFrame): The period table.
            time_matrix (pd.DataFrame): The time matrix.
            price_diff (pd.Series): The price differential.
            estimator (str): The estimator to use.
            **kwargs: Additional keyword arguments.

        Returns:
            LinearRegression: The repeat transaction house price model.

        """
        if len({len(repeat_trans_df), len(time_matrix), len(price_diff)}) > 1:
            msg = (
                "Number of observations of 'repeat_trans_df', 'time_matrix' and 'price_diff' do "
                "not match."
            )
            raise ValueError(msg)

        # Check for sparseness.
        if len(repeat_trans_df) < len(period_table):
            logging.info(
                f"You have fewer observations ({len(repeat_trans_df)}) than number of periods "
                f"({len(period_table)}). Results will likely be unreliable.",
            )

        args = (time_matrix, price_diff)
        if estimator == "base":
            return self._model_base(*args)
        if estimator == "robust":
            return self._model_robust(*args)
        if estimator == "weighted":
            return self._model_weighted(repeat_trans_df, *args, **kwargs)
        return None

    def _model_base(
        self,
        time_matrix: pd.DataFrame,
        price_diff: pd.Series,
    ) -> LinearRegression:
        """Fit a standard linear regression model.

        Args:
            time_matrix (pd.DataFrame): The time matrix.
            price_diff (pd.Series): The price differential.

        Returns:
            LinearRegression: The standard linear regression model.

        """
        return LinearRegression(fit_intercept=False).fit(time_matrix, price_diff)

    def _model_robust(
        self,
        time_matrix: pd.DataFrame,
        price_diff: pd.Series,
    ) -> LinearRegression:
        """Fit a robust (Huber) linear regression model.

        Args:
            time_matrix (pd.DataFrame): The time matrix.
            price_diff (pd.Series): The price differential.

        Returns:
            LinearRegression: The robust linear regression model.

        """
        # Determine 'sparseness' of the data.
        # time_size = np.median(
        #     pd.DataFrame(
        #         {
        #             "period_1": repeat_trans_df["period_1"],
        #             "period_2": repeat_trans_df["period_2"],
        #         }
        #     )
        #     .groupby(["period_1", "period_2"])
        #     .size()
        # )

        # TODO: Use different robust packages based on sparseness.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            return HuberRegressor(
                epsilon=1.345,
                max_iter=20,
                alpha=0,
                fit_intercept=False,
                tol=1e-4,
            ).fit(time_matrix, price_diff)

    def _model_weighted(
        self,
        repeat_trans_df: pd.DataFrame,
        time_matrix: pd.DataFrame,
        price_diff: pd.Series,
        **kwargs: Any,
    ) -> LinearRegression:
        """Fit a weighted linear regression model.

        Args:
            repeat_trans_df (pd.DataFrame): The repeat transaction data.
            time_matrix (pd.DataFrame): The time matrix.
            price_diff (pd.Series): The price differential.
            **kwargs: Additional keyword arguments.

        Returns:
            LinearRegression: The weighted linear regression model.

        """
        if "weights" not in kwargs:
            # Run base model.
            lm_model = LinearRegression(fit_intercept=False).fit(time_matrix, price_diff)

            # Estimate impact of time diff on errors.
            time_diff = pd.DataFrame(
                {"time_diff": repeat_trans_df["period_2"] - repeat_trans_df["period_1"]},
            )
            residuals = price_diff - lm_model.predict(time_matrix)
            err_fit = LinearRegression().fit(time_diff, residuals**2)

            # Implement weights.
            wgts = err_fit.predict(time_diff)
            wgts = np.where(wgts > 0, 1 / wgts, np.zeros(wgts.shape))
        else:
            wgts = kwargs["weights"]

        # Re-run model.
        return LinearRegression(fit_intercept=False).fit(
            time_matrix,
            price_diff,
            sample_weight=wgts,
        )

    def fit(
        self,
        estimator: str = "base",
        log_dep: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Fit the repeat transaction model and generate index coefficients.

        Args:
            estimator (str, optional): Estimator type. One of "base",
                "robust", or "weighted".
                Defaults to "base".
            log_dep (bool, optional): Log transform the dependent variable.
                Defaults to True.

        Returns:
            Self: The repeat transaction house price model.

        """
        # Create time matrix.
        time_matrix = self.create_time_matrix(self.hpi_df)

        # Calculate price differential.
        if log_dep:
            price_diff = np.log(self.hpi_df["price_2"]) - np.log(self.hpi_df["price_1"])
        else:
            price_diff = self.hpi_df["price_2"] - self.hpi_df["price_1"]

        # If any NA, NaN, or Inf/-Inf.
        if np.any(~np.isfinite(price_diff)):
            msg = "NA, negative, zero or non-finite values in the price field."
            raise ValueError(msg)

        # Extract base period mean price.
        base_price = np.mean(
            self.hpi_df["price_1"][self.hpi_df["period_1"] == min(self.hpi_df["period_1"])],
        )

        # Check for legal estimator type.
        if estimator not in ["base", "robust", "weighted"]:
            logging.warning(
                "Provided estimator type is not supported. Allowed estimators are: "
                "'base', 'robust' or 'weighted'. Defaulting to 'base'.",
            )
            estimator = "base"

        # Set estimator class, call method.
        model = self._create_model(
            repeat_trans_df=self.hpi_df,
            period_table=self.period_table,
            time_matrix=time_matrix,
            price_diff=price_diff,
            estimator=estimator,
            **kwargs,
        )

        # Check for successful model estimation.
        if not isinstance(model, LinearRegression | HuberRegressor):
            msg = "Model estimator was unsuccessful."
            raise ValueError(msg)

        # Create coefficient dataframe.
        model_df = pd.DataFrame(
            {
                "time": [
                    self.hpi_df["period_1"].min(),
                    *[int(x[5:]) for x in time_matrix.columns],
                ],
                "coefficient": [0, *model.coef_],
            },
        )

        self.coefficients = model_df
        self.model_obj = model
        self.base_price = base_price
        self.periods = self.period_table
        self.params = {
            "estimator": estimator,
            "log_dep": log_dep,
        }

        return self


class HedonicModel(BaseHousePriceModel):
    """Hedonic house price model."""

    def _create_model(
        self,
        X: pd.DataFrame | pd.Series,
        y: pd.Series,
        estimator: str,
        **kwargs: Any,
    ) -> LinearRegression:
        """Create a hedonic house price model.

        Args:
            X (pd.DataFrame | pd.Series): The independent variables.
            y (pd.Series): The dependent variable.
            estimator (str): The estimator to use.
            **kwargs: Additional keyword arguments.

        Returns:
            LinearRegression: The hedonic house price model.

        """
        return self._model_with_coefficients(X, y, estimator, **kwargs)

    def _model_with_coefficients(
        self,
        X: pd.DataFrame | pd.Series,
        y: pd.Series,
        estimator: str,
        **kwargs: Any,
    ) -> LinearRegression:
        """Fit model and populate coefficients to produce index.

        Args:
            X (pd.DataFrame | pd.Series): The independent variables.
            y (pd.Series): The dependent variable.
            estimator (str): The estimator to use.
            **kwargs: Additional keyword arguments.

        Returns:
            LinearRegression: The hedonic house price model.

        """
        if estimator == "base":
            return self._hed_model_base(X, y)
        if estimator == "robust":
            return self._hed_model_robust(X, y)
        if estimator == "weighted":
            return self._hed_model_weighted(X, y, **kwargs)
        return None

    def _hed_model_base(
        self,
        X: pd.DataFrame | pd.Series,
        y: pd.Series,
    ) -> LinearRegression:
        """Fit hedonic linear regression model.

        Args:
            X (pd.DataFrame | pd.Series): The independent variables.
            y (pd.Series): The dependent variable.

        Returns:
            LinearRegression: The hedonic linear regression model.

        """
        return LinearRegression().fit(X, y)

    def _hed_model_robust(
        self,
        X: pd.DataFrame | pd.Series,
        y: pd.Series,
    ) -> LinearRegression:
        """Fit robust hedonic linear regression model.

        Args:
            X (pd.DataFrame | pd.Series): The independent variables.
            y (pd.Series): The dependent variable.

        Returns:
            LinearRegression: The robust hedonic linear regression model.

        """
        # TODO: Use different robust packages based on sparseness.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            return HuberRegressor(
                epsilon=1.345,
                max_iter=20,
                alpha=0,
                tol=1e-4,
            ).fit(X, y)

    def _hed_model_weighted(
        self,
        X: pd.DataFrame | pd.Series,
        y: pd.Series,
        **kwargs: Any,
    ) -> LinearRegression:
        """Fit weighted hedonic linear regression model.

        Args:
            X (pd.DataFrame | pd.Series): The independent variables.
            y (pd.Series): The dependent variable.
            **kwargs: Additional keyword arguments.

        Returns:
            LinearRegression: The weighted hedonic linear regression model.

        """
        # Extract weights.
        wgts = kwargs["weights"]
        if len(wgts) != X.shape[0]:
            wgts = np.repeat(1, X.shape[0])
            logging.warning(
                "You have supplied a set of weights that do not match the data. "
                "Model being run in base OLS format.",
            )
        return LinearRegression().fit(X, y, sample_weight=wgts)

    def fit(
        self,
        estimator: str = "base",
        log_dep: bool = True,
        dep_var: str | None = None,
        ind_var: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the hedonic model and generate index coefficients.

        Args:
            estimator (str, optional): Estimator type. One of "base",
                "robust", or "weighted".
                Defaults to "base".
            log_dep (bool, optional): Log transform the dependent variable.
                Defaults to True.
            dep_var (str | None, optional): Dependent variable.
                Defaults to None.
            ind_var (list[str] | None, optional): Independent variable(s).
                Defaults to None.

        Returns:
            Self: The hedonic house price model.

        """
        hpi_df = self.hpi_df.copy()

        if dep_var is None or ind_var is None:
            msg = "'dep_var' and 'ind_var' must be supplied."
            raise ValueError(msg)

        oh_enc = OneHotEncoder(drop="first")
        oh_enc.fit(hpi_df[["trans_period"]])
        X_cats = pd.DataFrame(
            oh_enc.transform(hpi_df[["trans_period"]]).toarray(),
            columns=oh_enc.get_feature_names_out(["trans_period"]),
        )

        for var in ind_var:
            if hpi_df[var].dtype == "object":
                hpi_df[var] = hpi_df[var].astype("category")
                hpi_df[var] = hpi_df[var].cat.codes

        X = pd.concat([hpi_df[ind_var].reset_index(drop=True), X_cats], axis=1)
        y = np.log(hpi_df[dep_var]) if log_dep else hpi_df[dep_var]

        # Extract base period mean price.
        base_price = hpi_df["price"][hpi_df["trans_period"] == hpi_df["trans_period"].min()].mean()

        # Check for legal estimator type.
        if estimator not in ["base", "robust", "weighted"]:
            logging.warning(
                "Provided estimator type is not supported. Allowed estimators are: "
                "'base', 'robust' or 'weighted'.  Defaulting to 'base'.",
            )
            estimator = "base"

        # Check log dep vs data.
        if (log_dep and np.any(hpi_df["price"] <= 0)) or (hpi_df["price"].isnull().sum() > 0):
            msg = "Your 'price' field includes invalid values."
            raise ValueError(msg)

        # Set estimator class, call method.
        if estimator == "weighted" and "weights" not in kwargs:
            logging.warning(
                "You selected a weighted model but did not supply any weights. "
                "'weights' argument is NULL. Model run in base OLS format.",
            )
            estimator = "base"

        model = self._create_model(
            X=X,
            y=y,
            estimator=estimator,
            **kwargs,
        )

        if not isinstance(model, LinearRegression | HuberRegressor):
            msg = "Model estimator was unsuccessful."
            raise ValueError(msg)

        # Period names.
        p_names = [len(ind_var) + idx for idx in range(len(oh_enc.categories_[0]) - 1)]
        periods = list(range(1, len(oh_enc.categories_[0]) + 1))

        # Coefficients.
        coefs = np.hstack([[0], model.coef_[p_names]])

        model_df = pd.DataFrame({"time": periods, "coefficient": coefs})

        self.coefficients = model_df
        self.model_obj = model
        self.base_price = base_price
        self.periods = self.period_table
        self.params = {
            "estimator": estimator,
            "log_dep": log_dep,
            "ind_var": ind_var,
            "dep_var": dep_var,
        }

        return self
