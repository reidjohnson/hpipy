"""Random forest HPI extensions."""

import logging
from typing import Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence

from hpipy.price_index import HedonicIndex
from hpipy.price_model import BaseHousePriceModel


class RandomForestIndex(HedonicIndex):
    """Random forest house price index.

    Estimates the relationship between independent and dependent variables
    using a random forest model. Extracts the coefficients of temporal
    variables to represent the marginal contribution of each time period.
    Partial dependence is used to assess the conditional marginal impact of
    each time period on price changes, which are converted into an index.

    """

    @staticmethod
    def get_model() -> type[BaseHousePriceModel]:
        """Get a random forest house price model."""
        return RandomForestModel


class RandomForestModel(BaseHousePriceModel):
    """Random forest house price model."""

    def _create_model(
        self,
        X: pd.DataFrame | pd.Series,
        y: pd.Series,
        **kwargs: Any,
    ) -> RandomForestRegressor | RandomForestQuantileRegressor:
        """Create a random forest house price model.

        Args:
            X (pd.DataFrame | pd.Series): Independent variables.
            y (pd.Series): Dependent variable.
            **kwargs: Additional keyword arguments.

        Returns:
            RandomForestRegressor | RandomForestQuantileRegressor: Random forest model.

        """
        return self._model_with_coefficients(X, y, **kwargs)

    def _model_with_coefficients(
        self,
        X: pd.DataFrame | pd.Series,
        y: pd.Series,
        estimator: str,
        n_estimators: int,
        quantile: float | None,
        random_seed: int,
        **kwargs: Any,
    ) -> RandomForestRegressor | RandomForestQuantileRegressor:
        """Fit model and populate coefficients to produce index.

        Args:
            X (pd.DataFrame | pd.Series): Independent variables.
            y (pd.Series): Dependent variable.
            estimator (str): Estimator type.
            n_estimators (int): Number of estimators.
            quantile (float | None): Quantile to compute.
            random_seed (int): Random seed.
            **kwargs: Additional keyword arguments.

        Returns:
            RandomForestRegressor | RandomForestQuantileRegressor: Random forest model.

        """
        # Fit the model.
        cls = RandomForestQuantileRegressor if quantile is not None else RandomForestRegressor
        model = cls(
            n_estimators=n_estimators,
            # categorical_features=["trans_period"],
            random_state=random_seed,
        ).fit(X, y)
        if estimator == "pdp":
            return self._model_pdp(model, X, y, random_seed=random_seed, **kwargs)
        raise ValueError

    def _model_pdp(
        self,
        model: RandomForestRegressor | RandomForestQuantileRegressor,
        X: pd.DataFrame | pd.Series,
        y: pd.Series,
        log_dep: bool = False,
        random_seed: int = 0,
        **kwargs: Any,
    ) -> RandomForestRegressor | RandomForestQuantileRegressor:
        """Generate explanation from partial dependence plot (PDP) values.

        Args:
            model (RandomForestRegressor | RandomForestQuantileRegressor): Random forest model.
            X (pd.DataFrame | pd.Series): Independent variables.
            y (pd.Series): Dependent variable.
            log_dep (bool, optional): Log transform the dependent variable.
                Defaults to False.
            random_seed (int): Random seed.
            **kwargs: Additional keyword arguments.

        Returns:
            RandomForestRegressor | RandomForestQuantileRegressor: Random forest model.

        """
        # Get simulation dataframe.
        sim_X, _ = self._create_sim_df(X, y, random_seed=random_seed, **kwargs)

        # col_indices = [idx for idx, x in enumerate(X.columns) if x.startswith("trans_period_")]
        # partial_dependencies = []
        # for col_idx in range(len(X.columns)):
        #     predictions = partial_dependence(
        #         model, sim_X, col_idx, categorical_features=col_indices
        #     )
        #     partial_dependencies.append(predictions["average"].squeeze()[-1])

        col_idx = list(X.columns).index("trans_period")
        predictions = partial_dependence(model, sim_X, col_idx, categorical_features=[col_idx])
        values = predictions["grid_values"][0]
        partial_dependencies = predictions["average"].squeeze()

        pdp_df = (
            pd.DataFrame({"yhat": np.nan}, index=np.arange(X["trans_period"].max()) + 1)
            .combine_first(pd.DataFrame({"yhat": partial_dependencies}, index=values))
            .interpolate(fill_value="ffill")
            .bfill()
        )

        base_yhat = pdp_df["yhat"].iloc[0]

        # Add coefficients.
        coefs = pdp_df["yhat"] - base_yhat if log_dep else pdp_df["yhat"] / base_yhat

        model.coef_ = coefs

        return model

    def _create_sim_df(
        self,
        X: pd.DataFrame | pd.Series,
        y: pd.Series,
        random_seed: int,
        sim_ids: int | None = None,
        sim_count: int | None = None,
        sim_per: float | None = None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame | pd.Series, pd.Series]:
        """Create simulation data.

        Args:
            X (pd.DataFrame | pd.Series): Independent variables.
            y (pd.Series): Dependent variable.
            random_seed (int): Random seed.
            sim_ids (int | None, optional): Simulation IDs.
            sim_count (int | None, optional): Simulation count.
            sim_per (float | None, optional): Simulation percentage.

        Returns:
            tuple[pd.DataFrame | pd.Series, pd.Series]: Simulation data.

        """
        # If no filters.
        if sim_ids is None and sim_count is None and sim_per is None:
            return X, y

        # If by sim id.
        if sim_ids is not None:
            return X.iloc[sim_ids], y[sim_ids]

        # If a sim percentage is provided.
        if sim_count is None:
            sim_count = int(np.floor(sim_per * len(X))) if sim_per is not None else len(X)

        # Take sample.
        np.random.seed(random_seed)
        indices = np.random.randint(len(X), size=sim_count)
        X = X.iloc[indices[:sim_count]]
        y = y[indices[:sim_count].tolist()]

        return X, y

    def fit(
        self,
        dep_var: str | None = None,
        ind_var: list[str] | None = None,
        estimator: str = "pdp",
        log_dep: bool = True,
        n_estimators: int = 100,
        quantile: float | None = None,
        random_seed: int = 0,
        **kwargs: Any,
    ) -> Self:
        """Fit the random forest model and generate index coefficients.

        Args:
            dep_var (str | None, optional): Dependent variable.
                Defaults to None.
            ind_var (list[str] | None, optional): Independent variable(s).
                Defaults to None.
            estimator (str, optional): Estimator type.
                Defaults to "pdp".
            log_dep (bool, optional): Log transform the dependent variable.
                Defaults to True.
            n_estimators (int, optional): Number of estimators.
                Defaults to 100.
            quantile (float | None, optional): Quantile to compute.
                Defaults to None.
            random_seed (int, optional): Random seed to use.
                Defaults to 0.

        Returns:
            Self: Fitted model.

        """
        hpi_df = self.hpi_df.copy()

        if dep_var is None or ind_var is None:
            msg = "'dep_var' and 'ind_var' must be supplied."
            raise ValueError(msg)

        # oh_enc = OneHotEncoder(drop="first")
        # oh_enc.fit(hpi_df[["trans_period"]])
        # X_cats = pd.DataFrame(
        #     oh_enc.transform(hpi_df[["trans_period"]]).toarray(),
        #     columns=oh_enc.get_feature_names_out(["trans_period"]),
        # )
        # X = pd.concat([hpi_df[ind_var].reset_index(drop=True), X_cats], axis=1)

        for var in ind_var:
            if hpi_df[var].dtype == "object":
                hpi_df[var] = hpi_df[var].astype("category")
                hpi_df[var] = hpi_df[var].cat.codes

        X = hpi_df[[*ind_var, "trans_period"]].reset_index(drop=True)
        y = np.log(hpi_df[dep_var]) if log_dep else hpi_df[dep_var]

        # Extract base period mean price.
        base_price = hpi_df["price"][hpi_df["trans_period"] == hpi_df["trans_period"].min()].mean()

        # Check for legal estimator type.
        if estimator not in ["pdp"]:
            logging.warning(
                "Provided estimator type is not supported. Allowed estimators are: "
                "'pdp'. Defaulting to 'pdp'.",
            )
            estimator = "pdp"

        # Check log dep vs data.
        if (log_dep and np.any(hpi_df["price"] <= 0)) or (hpi_df["price"].isnull().sum() > 0):
            msg = "Your 'price' field includes invalid values."
            raise ValueError(msg)

        model = self._create_model(
            X=X,
            y=y,
            estimator=estimator,
            n_estimators=n_estimators,
            quantile=quantile,
            log_dep=log_dep,
            random_seed=random_seed,
            **kwargs,
        )

        # Check for successful model estimation.
        if not isinstance(model, (RandomForestRegressor | RandomForestQuantileRegressor)):
            msg = "Model estimator was unsuccessful."
            raise ValueError(msg)

        # Period names.
        # p_names = [len(ind_var) + idx for idx in range(len(oh_enc.categories_[0]) - 1)]
        # periods = list(range(1, len(oh_enc.categories_[0]) + 1))
        periods = list(np.arange(hpi_df["trans_period"].max()) + 1)

        # Coefficients.
        # coefs = np.hstack([[0], model.coef_[p_names]])
        coefs = model.coef_

        model_df = pd.DataFrame({"time": periods, "coefficient": coefs})

        self.coefficients = model_df
        self.model_obj = model
        self.periods = self.period_table
        self.base_price = base_price
        self.params = {
            "estimator": estimator,
            "log_dep": log_dep,
            "ind_var": ind_var,
            "dep_var": dep_var,
            "n_estimators": n_estimators,
            "random_seed": random_seed,
        }
        self.X = X
        self.y = y

        return self
