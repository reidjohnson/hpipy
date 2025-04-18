"""Neural network HPI extensions."""

import copy
import logging
from typing import Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta

from hpipy.extensions.neural.neural_avm.data_pipeline import DataPipeline
from hpipy.extensions.neural.neural_avm.data_preprocessors import (
    BaseFeaturePreprocessor,
    GeospatialPreprocessor,
    TemporalPreprocessor,
)
from hpipy.extensions.neural.neural_avm.data_transformers import (
    FeatureTransformer,
    ResponseTransformer,
)
from hpipy.extensions.neural.neural_avm.model_fnn import NeuralAVM
from hpipy.price_index import HedonicIndex
from hpipy.price_model import BaseHousePriceModel


class NeuralAVMWithCoef(NeuralAVM):
    """NeuralAVM with coefficients."""

    coef_: np.ndarray


class NeuralNetworkIndex(HedonicIndex):
    """Neural network house price index.

    Estimates the relationship between independent variables and the dependent
    variable using a neural network model. The index can be derived from the
    model using two distinct approaches:

    1. Residual Approach: This method extracts the house price index directly
       from the market pathway of the neural network. It isolates the temporal
       component of price changes by zeroing out all non-time features and
       examining the network's output, effectively capturing the "residual"
       market trend.

    2. Attributional Approach: This method derives the index by analyzing the
       explainability of both market and time components. It uses attribution
       techniques to decompose the network's predictions and to assess the
       conditional marginal impact of each time period on price changes, which
       are converted into an index.

    Both approaches aim to capture market trends, but differ in how they extract
    temporal information from the neural network's learned representations.

    """

    @staticmethod
    def get_model() -> type[BaseHousePriceModel]:
        """Get a neural network house price model."""
        return NeuralNetworkModel


class NeuralNetworkModel(BaseHousePriceModel):
    """Neural network house price model."""

    def _create_model(
        self,
        X: pd.DataFrame | pd.Series,
        y: pd.Series,
        **kwargs: Any,
    ) -> NeuralAVMWithCoef:
        """Create a neural network house price model.

        Args:
            X (pd.DataFrame | pd.Series): Input data.
            y (pd.Series): Target data.
            **kwargs: Additional keyword arguments.

        Returns:
            NeuralAVMWithCoef: Neural network house price model.

        """
        return self._model_with_coefficients(X, y, **kwargs)

    def _model_with_coefficients(
        self,
        X: pd.DataFrame | pd.Series,
        y: pd.Series,
        estimator: str,
        partition_cols: list | None = None,
        **kwargs: Any,
    ) -> NeuralAVMWithCoef:
        """Fit model and populate coefficients to produce index.

        Args:
            X (pd.DataFrame | pd.Series): Input data.
            y (pd.Series): Target data.
            estimator (str): Estimator type.
            partition_cols (list | None, optional): Partition columns.
            **kwargs: Additional keyword arguments.

        Returns:
            NeuralAVMWithCoef: Neural network house price model.

        """
        model = NeuralAVMWithCoef(**kwargs["init"]).fit(X, y, **kwargs["fit"])
        periods = sorted(self.hpi_df["trans_period"].unique())
        if partition_cols is None:
            if estimator == "residual":
                model = self._model_residual(model, **kwargs["predict"])
            elif estimator == "attributional":
                model = self._model_attributional(model, **kwargs["predict"])
            else:
                raise ValueError
            model_df = pd.DataFrame({"time": periods, "coefficient": model.coef_})
            model.coef_ = model_df
            return model
        input_df = copy.deepcopy(kwargs["predict"]["input_df"])
        coef = []
        unique_df = input_df[partition_cols].drop_duplicates()
        for _, row in unique_df.iterrows():
            input_df_i = input_df.merge(row.to_frame().T, on=partition_cols, how="inner")
            kwargs["predict"]["input_df"] = input_df_i
            if estimator == "residual":
                model_i = self._model_residual(model, **kwargs["predict"])
            elif estimator == "attributional":
                model_i = self._model_attributional(model, **kwargs["predict"])
            else:
                raise ValueError
            model_i_df = pd.DataFrame({"time": periods, "coefficient": model_i.coef_})
            for col in partition_cols:
                model_i_df[col] = row[col][0]
            coef.append(model_i_df)
        model.coef_ = pd.concat(coef) if len(coef) > 1 else coef[0]
        return model

    def _create_explainer_input_data(
        self,
        model: NeuralAVMWithCoef,
        input_df: pd.DataFrame,
        date: str,
        data_pipeline: DataPipeline,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create explainer input for each date in the period table range.

        Args:
            model (NeuralAVMWithCoef): Neural network house price model.
            input_df (pd.DataFrame): Input data.
            date (str): Date column name.
            data_pipeline (DataPipeline): Data pipeline.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Explainable input data and
                transformed input data.

        """
        min_pred_datetime = self.period_table["start_date"].min()
        max_pred_datetime = self.period_table["end_date"].max()
        total_days = (max_pred_datetime - min_pred_datetime).days

        dfs = []
        for j in range(total_days + 1):
            df_i = input_df.iloc[:1, :].copy()
            df_i[date] = (min_pred_datetime + relativedelta(days=+j)).strftime("%Y-%m-%d")
            dfs.append(df_i)
        explain_df = pd.concat(dfs).reset_index(drop=True)

        X_input = data_pipeline.predict_transform(explain_df, override_date=False, return_y=False)

        # Deduplicate.
        inputs = model.prepare_model_input(X_input)
        for key in inputs:
            inputs[key] = torch.from_numpy(inputs[key].numpy().squeeze()[-len(X_input) :])
        input_df = pd.DataFrame(inputs).drop_duplicates()
        explain_df = explain_df.loc[input_df.index].reset_index(drop=True)
        X_input = data_pipeline.predict_transform(explain_df, override_date=False, return_y=False)

        return explain_df, X_input

    def _explainer_output_to_coef(self, coef_df: pd.DataFrame, date: str) -> pd.Series:
        """Convert explainer output to index coefficients.

        Args:
            coef_df (pd.DataFrame): Coefficient data.
            date (str): Date column name.

        Returns:
            pd.Series: Index coefficients.

        """
        # Match coefficients with their corresponding periods.
        for _, row in self.period_table.iterrows():
            after_start = coef_df[date] >= row["start_date"]
            before_end = coef_df[date] <= row["end_date"]
            coef_df.loc[after_start & before_end, "period"] = row["period"]

        # Calculate average value for each period and filter to needed periods.
        period_df = pd.DataFrame({"period": self.hpi_df["trans_period"].unique()})
        coef_df = (
            coef_df.groupby("period")
            .mean()
            .reset_index()
            .rename(columns={0: "yhat"})
            .merge(period_df, how="right")
            .drop(columns=["period", date])
        )

        # Normalize coefficients by baseline value.
        base_coef = coef_df["yhat"].iloc[0]
        return coef_df["yhat"] - base_coef

    @staticmethod
    def _model_inputs_to_features(
        data_pipeline: DataPipeline,
        inputs: str | list[str],
    ) -> list[str]:
        """Get transformed feature names for input feature columns.

        Args:
            data_pipeline (DataPipeline): The data pipeline.
            inputs (str | list[str]): The input feature columns.

        Returns:
            list[str]: The transformed feature names.

        """
        features = [inputs] if not isinstance(inputs, list) else inputs
        if data_pipeline.feature_preprocessors_ is not None:
            for feature in inputs:
                for fp in data_pipeline.feature_preprocessors_:
                    if feature in list(fp.input_cols.values()):
                        new_features = list({f["name"] for f in fp.output_cols})
                        features.remove(feature)
                        features.extend(new_features)
        return features

    def _model_residual(
        self,
        model: NeuralAVMWithCoef,
        input_df: pd.DataFrame,
        date: str,
        data_pipeline: DataPipeline,
        min_pred_epoch: int | None,
        quantile: float = 0.5,
        partition_cols: list | None = None,
        **kwargs: Any,
    ) -> NeuralAVMWithCoef:
        """Generate explanation from index-pathway output component.

        Args:
            model (NeuralAVMWithCoef): Neural network house price model.
            input_df (pd.DataFrame): Input data.
            date (str): Date column name.
            data_pipeline (DataPipeline): Data pipeline.
            min_pred_epoch (int | None): Minimum prediction epoch.
            quantile (float, optional): Quantile to estimate.
            partition_cols (list | None, optional): Partition columns.
            **kwargs: Additional keyword arguments.

        Returns:
            NeuralAVMWithCoef: Neural network house price model.

        """
        input_features = [date]
        if partition_cols is not None:
            input_features += partition_cols

        explain_df, X_input = self._create_explainer_input_data(
            model,
            input_df,
            date,
            data_pipeline,
        )

        features = self._model_inputs_to_features(data_pipeline, input_features)
        X_input[list(set(X_input.columns).difference(features))] = 0

        # Use index-pathway outputs on synthetic data as the coefficients.
        _, X2 = model.predict(
            X_input,
            quantiles=quantile,
            min_epoch=min_pred_epoch,
            max_epoch=None,
            **kwargs,
        )
        X2 = X2[-len(X_input) :, :]

        coef_df = (
            pd.DataFrame(X2)
            .pipe(lambda x: pd.concat([explain_df[[date]], x], axis=1))
            .assign(**{date: lambda x: pd.to_datetime(x[date], format="%Y-%m-%d")})
        )

        model.coef_ = self._explainer_output_to_coef(coef_df, date)

        return model

    def _model_attributional(
        self,
        model: NeuralAVMWithCoef,
        input_df: pd.DataFrame,
        date: str,
        data_pipeline: DataPipeline,
        min_pred_epoch: int | None,
        quantile: float = 0.5,
        partition_cols: list | None = None,
        **kwargs: Any,
    ) -> NeuralAVMWithCoef:
        """Generate explanation from attribution of property estimate.

        Args:
            model (NeuralAVMWithCoef): Neural network house price model.
            input_df (pd.DataFrame): Input data.
            date (str): Date column name.
            data_pipeline (DataPipeline): Data pipeline.
            min_pred_epoch (int | None): Minimum prediction epoch.
            quantile (float, optional): Quantile to estimate.
            partition_cols (list | None, optional): Partition columns.
            **kwargs: Additional keyword arguments.

        Returns:
            NeuralAVMWithCoef: Neural network house price model.

        """
        num_background = 1
        input_features = [date]
        if partition_cols is not None:
            input_features += partition_cols

        explain_df, X_input = self._create_explainer_input_data(
            model,
            input_df,
            date,
            data_pipeline,
        )

        X_baseline = data_pipeline.predict_transform(
            input_df.iloc[:num_background, :],
            override_date=False,
            return_y=False,
        )

        features = self._model_inputs_to_features(data_pipeline, input_features)

        X_input[list(set(X_input.columns).difference(features))] = 0
        X_baseline[list(set(X_baseline.columns).difference(features))] = 0  # type: ignore

        df_attrs = model.explain(
            X_input,
            X_baseline,
            quantile=quantile,
            min_epoch=min_pred_epoch,
            **kwargs,
        )

        coef_df = (
            df_attrs.loc[:, [*features, "_quantile"]]
            .pipe(lambda x: pd.concat([explain_df[date], x], axis=1))
            .set_index(date)
            .sum(axis=1)
            .reset_index()
            .assign(**{date: lambda x: pd.to_datetime(x[date], format="%Y-%m-%d")})
        )

        model.coef_ = self._explainer_output_to_coef(coef_df, date)

        return model

    def _get_data_transformers(
        self,
        X: pd.DataFrame,
        dep_var: str,
        date: str,
        feature_dict: dict[str, list[str]],
        preprocess_time: bool,
        preprocess_geo: bool,
        geo_resolutions: int | list[int],
    ) -> tuple[DataPipeline, FeatureTransformer]:
        """Get data transformers.

        Args:
            X (pd.DataFrame): Input data.
            dep_var (str): Dependent variable.
            date (str): Date column name.
            feature_dict (dict[str, list[str]]): Feature dictionary.
            preprocess_time (bool): Preprocess time.
            preprocess_geo (bool): Preprocess geospatial.
            geo_resolutions (int | list[int]): Geospatial resolutions.

        Returns:
            tuple[DataPipeline, FeatureTransformer]: Data pipeline and feature
                transformer.

        """
        # Check data.
        if np.any(self.hpi_df["price"] <= 0) or (self.hpi_df["price"].isnull().sum() > 0):
            msg = "Your 'price' field includes invalid values."
            raise ValueError(msg)

        min_train_datetime = X[date].min()
        max_train_datetime = X[date].max()

        feature_preprocessors: list[BaseFeaturePreprocessor] = []
        if preprocess_geo:
            if {"latitude", "longitude"}.issubset(set(X.columns)):
                feature_preprocessors += [
                    GeospatialPreprocessor(
                        resolutions=geo_resolutions,
                        latitude_col="latitude",
                        longitude_col="longitude",
                    ),
                ]
            else:
                logging.warning(
                    "Cannot preprocess geospatial features because 'latitude' and/or 'longitude' "
                    "columns are missing. Skipping geospatial preprocessing.",
                )
        if preprocess_time:
            if date in list(X.columns):
                feature_preprocessors += [
                    TemporalPreprocessor(
                        min_train_datetime.strftime(format="%Y-%m-%d"),
                        max_train_datetime.strftime(format="%Y-%m-%d"),
                        saledate_col=date,
                    ),
                ]
            else:
                logging.warning(
                    "Cannot preprocess temporal features because 'date' column is missing. "
                    "Skipping temporal preprocessing.",
                )

        feature_transformer = FeatureTransformer(feature_dict)
        response_transformer = ResponseTransformer()

        data_pipeline = DataPipeline(
            feature_dict,
            dep_var,
            feature_preprocessors=feature_preprocessors,
            feature_transformer=feature_transformer,
            response_transformer=response_transformer,
        )

        return data_pipeline, feature_transformer

    def fit(
        self,
        dep_var: str,
        ind_var: list[str],
        date: str,
        estimator: str = "residual",
        log_dep: bool = True,
        feature_dict: dict[str, list[str]] | None = None,
        num_models: int = 5,
        num_epochs: int = 20,
        batch_size: int = 1024,
        hidden_dims: int | list[int] | None = None,
        emb_size: int = 5,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        do_mixup: bool = False,
        preprocess_time: bool = True,
        preprocess_geo: bool = True,
        geo_resolutions: int | list[int] | None = None,
        min_pred_epoch: int = 16,
        quantile: float = 0.5,
        random_seed: int = 0,
        verbose: bool = False,
    ) -> Self:
        """Fit the neural network model and generate index coefficients.

        The model supports two distinct approaches for extracting the house
        price index, controlled by the `estimator` parameter:

        1. "residual": This approach extracts the index directly from the
           market pathway of the neural network. It works by zeroing out all
           of the neural network. It works by zeroing out all non-temporal
           features and examining the network's output, effectively isolating
           the temporal component of price changes. This method is more direct
           and computationally efficient.

        2. "attributional": This approach derives the index through
           explainability analysis of both market and time components. It uses
           attribution techniques to decompose the network's predictions and
           quantify how much of the price change can be attributed to temporal
           factors versus other market characteristics. This method provides
           more detailed insights into price drivers but is computationally
           more intensive.

        Args:
            dep_var (str): Dependent variable.
            ind_var (list[str]): Independent variable(s).
            date (str): Date column name.
            estimator (str, optional): Estimator type. Choose between
                "residual" (extracts index from market pathway) or
                "attributional" (derives index  through explainability).
                Defaults to "residual".
            log_dep (bool, optional): Log transform the dependent variable.
                Defaults to True.
            feature_dict (dict[str, list[str]] | None, optional): Feature
                dictionary.
                Defaults to None.
            num_models (int, optional): Number of models in the ensemble.
                Defaults to 5.
            num_epochs (int, optional): Number of epochs to train.
                Defaults to 20.
            batch_size (int, optional): Batch size for training.
                Defaults to 1024.
            hidden_dims (int | list[int] | None, optional): Hidden
                layer sizes.
                If None, defaults to [128, 32].
            emb_size (int, optional): Output size for each embedding.
                Defaults to 5.
            dropout_rate (float, optional): Dropout rate for training.
                Defaults to 0.1.
            learning_rate (float, optional): Learning rate for training.
                Defaults to 1e-3.
            do_mixup (bool, optional): Perform mixup augmentation.
                Defaults to False.
            preprocess_time (bool, optional): Preprocess temporal features
                from sale date column.
                Defaults to True.
            preprocess_geo (bool, optional): Preprocess geospatial features
                from latitude and longitude columns.
                Defaults to True.
            geo_resolutions (int | list[int] | None, optional):
                Geospatial (H3) cell resolutions.
                If None, defaults to [6, 7].
            min_pred_epoch (int, optional): Minimum prediction epoch.
                If None, defaults to 16.
            quantile (float, optional): Quantile to estimate.
                If None, defaults to 0.5.
            random_seed (int, optional): Random seed to use.
                Defaults to 0.
            verbose (bool, optional): Verbose output.
                Defaults to False.

        Returns:
            Self: Fitted model.

        """
        if not log_dep:
            raise NotImplementedError

        if feature_dict is None:
            feature_dict = {"log_numerics": ind_var, "hpi": [date]}

        if hidden_dims is None:
            hidden_dims = [128, 32]
        elif not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]

        if geo_resolutions is None:
            geo_resolutions = [6, 7]
        elif not isinstance(geo_resolutions, list):
            geo_resolutions = [geo_resolutions]

        X = self.hpi_df[[*ind_var, "trans_period", date]].reset_index(drop=True)
        y = self.hpi_df[dep_var].reset_index(drop=True)

        data_pipeline, feature_transformer = self._get_data_transformers(
            X,
            dep_var,
            date,
            feature_dict,
            preprocess_time,
            preprocess_geo,
            geo_resolutions,
        )

        train_df = X
        train_df[dep_var] = y

        X_train, y_train = data_pipeline.train_transform(train_df)

        feature_dict_trans = data_pipeline.get_transformed_feature_dict()
        init_dict = feature_transformer.prepare_init_dict(X_train)

        partition_cols = None
        if len(feature_dict["hpi"]) > 1:
            partition_cols = [col for col in feature_dict["hpi"] if col != date]

        # Check for legal estimator type.
        if estimator not in ["residual", "attributional"]:
            logging.warning(
                "Provided estimator type is not supported. Allowed estimators are: "
                "'residual' and 'attributional'. Defaulting to 'residual'.",
            )
            estimator = "residual"

        config = {
            "init": {
                "num_models": num_models,
                "init_dict": init_dict,
                "feature_dict": feature_dict_trans,
                "hidden_dims": hidden_dims,
                "emb_size": emb_size,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "verbose": verbose,
                "random_seed": random_seed,
            },
            "fit": {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "do_mixup": do_mixup,
                "verbose": verbose,
            },
            "predict": {
                "input_df": train_df,
                "date": date,
                "data_pipeline": data_pipeline,
                "min_pred_epoch": min_pred_epoch,
                "quantile": quantile,
                "partition_cols": partition_cols,
            },
        }

        model = self._create_model(
            X_train,
            y_train,
            estimator=estimator,
            partition_cols=partition_cols,
            **config,
        )

        self.coefficients = model.coef_
        self.model_obj = model
        self.periods = self.period_table
        self.base_price = 1
        self.params = {
            "ind_var": ind_var,
            "dep_var": dep_var,
            "date": date,
            "estimator": estimator,
            "log_dep": log_dep,
            "feature_dict": feature_dict,
            "num_models": num_models,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "hidden_dims": hidden_dims,
            "emb_size": emb_size,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "do_mixup": do_mixup,
            "preprocess_time": preprocess_time,
            "preprocess_geo": preprocess_geo,
            "geo_resolutions": geo_resolutions,
            "min_pred_epoch": min_pred_epoch,
            "quantile": quantile,
            "random_seed": random_seed,
            "verbose": verbose,
        }
        self.X = X
        self.y = y

        return self
