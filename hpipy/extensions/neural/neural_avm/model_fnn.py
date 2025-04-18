"""Neural network AVM for jointly estimating property valuations and HPIs.

The module structure is the following:

- The ``BaseNeuralAVM`` base class implements ``fit`` and ``predict` methods
  for training and predicting with an individual neural network.

- The ``NeuralAVM`` class implements ``fit`` and ``predict` methods for
  training and predicting with an ensemble of identical neural networks, each
  trained with a different seed and with their predictions averaged. It also
  implements an ``explain`` method for post-hoc feature attributions.
"""

import datetime
import logging
import os
import random
import shutil
import time
import warnings
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import DeepLift, DeepLiftShap
from torch import nn
from torch.utils.data import DataLoader

from .data_loader import TabularDataset, collate_fn
from .model_layers import MonotonicDense, OrdinalEmbedding
from .utils.data import prepare_dataframe, prepare_tensor
from .utils.error import NotFittedError
from .utils.model import get_device, mixup, quantile_loss

CURRENT_TIME = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

SAVE_PATH = "../model_outputs/neural_avm/" + CURRENT_TIME
SAVE_FORMAT = "model_{model_num:03d}_epoch_{epoch:03d}.pt"


class BaseNeuralAVM(nn.Module):
    """Base neural network AVM for estimating property valuations and indices.

    The model creates separate pathways for learning property-level effects
    and time/region (index) effects, which are additively combined to estimate
    (log) property values. The learned time/region component can be separately
    used as a house price index (HPI). The model minimizes the quantile loss.

    In order to estimate arbitrary quantiles, the model uses simultaneous
    quantile regression [1], where random quantile levels are sampled during
    training and used as an input to the model and loss function.

    The model employs monotonic dense layers [2] to structurally constrain the
    monotonicity of the quantile predictions and prohibit crossing quantiles.

    References:
        [1] Tagasovska, N., and D. Lopez-Paz. "Single-Model Uncertainties for
            Deep Learning." Advances in Neural Information Processing Systems,
            32, 2019.
        [2] Runje, D., and S. M. Shankaranarayana. "Constrained Monotonic
            Neural Networks." Proceedings of the International Conference on
            Machine Learning. PMLR, 2023.

    """

    def __init__(
        self,
        init_dict: dict[str, np.ndarray],
        feature_dict: dict[str, list[str]],
        hidden_dims: list[int],
        emb_size: int,
        dropout_rate: float,
        learning_rate: float,
    ) -> None:
        """Initialize the model.

        Args:
            init_dict (dict[str, np.ndarray]): Initialization dictionary.
            feature_dict (dict[str, list[str]]): Feature dictionary.
            hidden_dims (list[int], optional): Hidden layer sizes.
            emb_size (int): Output size for each embedding.
            dropout_rate (float): Dropout rate for training.
            learning_rate (float): Learning rate for training.

        """
        super().__init__()

        input_size_prp, input_size_hpi, x_emb = self._define_inputs(
            init_dict,
            feature_dict,
            emb_size,
        )
        dense_prp, dense_hpi = self._define_dense(
            input_size_prp,
            input_size_hpi,
            hidden_dims,
            dropout_rate,
        )

        self.x_emb = nn.ModuleDict(x_emb)
        self.dense_prp = nn.Sequential(*dense_prp)
        self.dense_hpi = nn.Sequential(*dense_hpi)

        params = [
            {"params": self.x_emb.parameters()},
            {"params": self.dense_prp.parameters()},
            {"params": self.dense_hpi.parameters()},
        ]
        self.optimizer = self._optimizer(params, learning_rate)

        self.feature_dict = feature_dict
        self.dropout_rate = dropout_rate
        self.input_size_prp = input_size_prp
        self.input_size_hpi = input_size_hpi

    def _define_inputs(
        self,
        init_dict: dict[str, np.ndarray],
        feature_dict: dict[str, list[str]],
        emb_size: int,
    ) -> tuple[int, int, dict[str, nn.Module]]:
        """Define the input layer.

        Args:
            init_dict (dict[str, np.ndarray]): Initialization dictionary.
            feature_dict (dict[str, list[str]]): Feature dictionary.
            emb_size (int): Embedding size.

        Returns:
            tuple[int, int, dict[str, nn.Module]]: Input size, input size,
                and embedding dictionary.

        """
        input_size_prp = 0
        input_size_hpi = 0
        x_emb: dict[str, nn.Module] = {}

        for key in init_dict:
            if (
                key in feature_dict["nulls"]
                or key in feature_dict["numerics"]
                or key in feature_dict["log_numerics"]
            ):
                input_n_size = 1
            elif key in feature_dict["categoricals"]:
                num_embeddings = int(init_dict[key].max() + 1)
                emb = nn.Embedding(num_embeddings, emb_size)
                x_emb[key] = emb
                input_n_size = emb_size
            elif key in feature_dict["ordinals"]:
                num_embeddings = int(init_dict[key].max())
                ord_emb = OrdinalEmbedding(num_embeddings, emb_size)
                x_emb[key] = ord_emb
                input_n_size = emb_size
            else:
                continue

            input_size_prp += input_n_size
            if "hpi" in feature_dict and key in feature_dict["hpi"]:
                input_size_hpi += input_n_size

        return input_size_prp, input_size_hpi, x_emb

    def _define_dense(
        self,
        input_size_prp: int,
        input_size_hpi: int,
        hidden_dims: list[int],
        dropout_rate: float,
    ) -> tuple[list[nn.Module], list[nn.Module]]:
        """Define the dense layers.

        Args:
            input_size_prp (int): Input size for property-level dense layers.
            input_size_hpi (int): Input size for index-level dense layers.
            hidden_dims (list[int]): List of hidden layer sizes.
            dropout_rate (float): Dropout rate.

        Returns:
            tuple[list[nn.Module], list[nn.Module]]: Property-level and
                index-level dense layers.

        """
        # Apply monotonicity constraint to quantiles.
        # Quantiles are concatenated to the final layer input.
        monotonicity_indicator = ([0] * hidden_dims[-1]) + [1]

        # Define the property-level dense layers.
        dense_prp: list[nn.Module] = []
        sizes_prp = [input_size_prp, *hidden_dims, 1]
        for i, (n_in, n_out) in enumerate(itertools.pairwise(sizes_prp)):
            if i == 0 and dropout_rate > 0:
                # Use dropout to bias the learning to index-level layers.
                dense_prp.append(nn.Dropout(dropout_rate if input_size_hpi > 0 else 0))
            if i + 2 < len(sizes_prp):
                dense_prp.append(nn.Linear(n_in, n_out))
                dense_prp.append(nn.ReLU())
            else:
                dense_prp.append(
                    MonotonicDense(
                        n_in + 1,  # account for quantile input
                        n_out,
                        activation=None,
                        monotonicity_indicator=monotonicity_indicator,
                    ),
                )

        # Define the index-level dense layers.
        dense_hpi: list[nn.Module] = []
        if input_size_hpi > 0:
            sizes_hpi = [input_size_hpi, *hidden_dims, 1]
            for i, (n_in, n_out) in enumerate(itertools.pairwise(sizes_hpi)):
                if i + 2 < len(sizes_hpi):
                    dense_hpi.append(nn.Linear(n_in, n_out))
                    dense_hpi.append(nn.ReLU())
                else:
                    dense_hpi.append(
                        MonotonicDense(
                            n_in + 1,  # account for quantile input
                            n_out,
                            activation=None,
                            monotonicity_indicator=monotonicity_indicator,
                        ),
                    )

        return dense_prp, dense_hpi

    def _optimizer(
        self,
        params: list[dict[str, Iterator[nn.Parameter]]],
        learning_rate: float,
    ) -> torch.optim.Optimizer:
        """Initialize the optimizer.

        Args:
            params (list[dict[str, Iterator[nn.Parameter]]]): Parameters to
                optimize.
            learning_rate (float): Learning rate.

        Returns:
            torch.optim.Optimizer: Optimizer.

        """
        return torch.optim.Adam(params, lr=learning_rate)

    def _prepare_quantiles(
        self,
        x: torch.Tensor,
        quantiles: torch.Tensor | list[float] | None = None,
    ) -> torch.Tensor:
        """Check, scale, and concatenate quantiles with inputs.

        Args:
            x (torch.Tensor): Input tensor.
            quantiles (torch.Tensor | list[float] | None, optional):
                Quantiles.
                Defaults to None.

        Returns:
            torch.Tensor: Concatenated tensor.

        """
        if quantiles is None:
            quantiles = torch.Tensor([0.5])
        elif not isinstance(quantiles, torch.Tensor):
            quantiles = torch.Tensor(quantiles)
        quantiles = quantiles.to(x.device)
        if quantiles.nelement() == 1:  # extend single quantile to entire batch
            quantiles = quantiles.repeat(x.size(0), 1)
        elif len(quantiles.size()) == 1:  # unsqueeze a batch of quantiles
            quantiles = quantiles.unsqueeze(-1)
        quantiles = (quantiles - 0.5) * 2  # scale to the range [-1, 1]
        return torch.cat([x, quantiles], 1)

    def _forward_inputs(
        self,
        x_dict: dict[str, torch.Tensor],
        dropout_rate: float | None,
        training: bool,
    ) -> torch.Tensor:
        """Forward pass through the input layer.

        Args:
            x_dict (dict[str, torch.Tensor]): Input dictionary.
            dropout_rate (float | None): Dropout rate.
            training (bool): Training flag.

        Returns:
            torch.Tensor: Concatenated property and index pathways tensor.

        """
        x_dict = {k: v.to(get_device()) for k, v in x_dict.items()}

        x_prp = torch.Tensor().to(get_device())
        x_hpi = torch.Tensor().to(get_device())

        for key in x_dict:
            x = x_dict[key]
            if (
                key in self.feature_dict["nulls"]
                or key in self.feature_dict["numerics"]
                or key in self.feature_dict["log_numerics"]
            ):
                xi = x.float()
            elif key in self.feature_dict["categoricals"] or key in self.feature_dict["ordinals"]:
                xi = x.long()
                xi = torch.flatten(self.x_emb[key](xi), 1, -1)
            else:
                continue

            if xi is not None and xi.ndim == 1:
                xi = xi.reshape(-1, 1)

            if "hpi" in self.feature_dict and key in self.feature_dict["hpi"]:
                x_hpi = torch.cat([x_hpi, xi], 1)
                if dropout_rate is None:
                    dropout_rate = self.dropout_rate
                if dropout_rate > 0:
                    # Use dropout to bias the learning to index-level layers.
                    xi = F.dropout(xi, p=dropout_rate, training=training)

            x_prp = torch.cat([x_prp, xi], 1)

        return torch.cat([x_prp, x_hpi], dim=1)

    def _forward_dense(
        self,
        x_prp: torch.Tensor,
        x_hpi: torch.Tensor,
        quantiles: torch.Tensor | list[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the dense layers.

        Args:
            x_prp (torch.Tensor): Property-level input tensor.
            x_hpi (torch.Tensor): Index-level input tensor.
            quantiles (torch.Tensor | list[float] | None, optional):
                Quantiles.
                Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Property and index pathways
                tensors.

        """
        # Property-level (log) effects.
        for i, layer in enumerate(self.dense_prp):
            if i + 1 == len(self.dense_prp):
                x_prp = self._prepare_quantiles(x_prp, quantiles)
            x_prp = layer(x_prp)

        # Index-level (log) effects.
        if len(self.dense_hpi) > 0:
            for i, layer in enumerate(self.dense_hpi):
                if i + 1 == len(self.dense_hpi):
                    x_hpi = self._prepare_quantiles(x_hpi, quantiles)
                x_hpi = layer(x_hpi)
        else:
            x_hpi = torch.zeros_like(x_prp)

        return x_prp, x_hpi

    def forward(
        self,
        x: dict[str, torch.Tensor] | torch.Tensor,
        _: Any = None,
        quantiles: torch.Tensor | list[float] | None = None,
        dropout_rate: float | None = None,
        training: bool = False,
        return_hpi: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass.

        Args:
            x (dict[str, torch.Tensor] | torch.Tensor): Inputs.
            _ (Any, optional): Hook for arguments to custom methods.
                Defaults to None.
            quantiles (torch.Tensor | list[float] | None, optional):
                Quantiles to estimate.
                Defaults to None
            dropout_rate (float | None, optional): Dropout rate.
                Defaults to None.
            training (bool, optional): Training Boolean.
                Defaults to False.
            return_hpi (bool, optional): Return HPI component Boolean.
                Defaults to False.

        Returns:
            torch.Tensor: Estimated property value (and index component).

        """
        if isinstance(x, dict):
            x_in = self._forward_inputs(x, dropout_rate, training)
        elif isinstance(x, torch.Tensor):
            if quantiles is None:
                x_in = x[:, :-1]
                quantiles = x[:, -1]
            else:
                x_in = x
        else:
            msg = f"{type(x)}"
            raise ValueError(msg)

        # Separate the property and index pathway inputs.
        x_prp = x_in[:, : self.input_size_prp]
        x_hpi = x_in[:, self.input_size_prp : self.input_size_prp + self.input_size_hpi]

        x_prp, x_hpi = self._forward_dense(x_prp, x_hpi, quantiles)

        x_out = x_prp + x_hpi

        return (x_out, x_hpi) if return_hpi else x_out

    def fit(
        self,
        train_gen: DataLoader,
        save_path: str,
        num_epochs: int,
        model_num: int = 0,
        dropout_rate: float | None = None,
        do_mixup: bool = False,
        loss_fn: Callable[..., torch.Tensor] = quantile_loss,
        verbose: bool = True,
    ) -> None:
        """Fit the model.

        Args:
            train_gen (DataLoader): Training data loader.
            save_path (str): Model save path.
            num_epochs (int): Number of epochs to train.
            model_num (int): Model number, if in an ensemble.
                Defaults to 0.
            dropout_rate (float | None, optional): Dropout rate.
                Defaults to None.
            do_mixup (bool, optional): Perform mixup augmentation.
                Defaults to False.
            loss_fn (Callable[..., torch.Tensor], optional): Loss function.
                Defaults to quantile_loss function.
            verbose (bool, optional): Verbose output.
                Defaults to True.

        """
        os.makedirs(save_path, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            start = time.time()
            losses = []
            for batch in train_gen:
                if isinstance(batch.x, torch.Tensor):
                    X_dict = prepare_tensor(batch.x, self.feature_dict)
                else:
                    X_dict = {k: v.to(get_device()) for k, v in batch.x.items()}
                y = batch.y.to(get_device())
                if y.ndim == 0:
                    y = y.view(1, 1)

                # Generate a batch of random quantiles for training input.
                quantiles = torch.rand(batch.y.size(0)).to(get_device())

                X = self._forward_inputs(X_dict, dropout_rate, True)

                if do_mixup:
                    X, y_a, y_b, lam = mixup(X, y, alpha=0.1)

                y_pred = self.forward(  # type: ignore
                    X,
                    batch,
                    quantiles,
                    dropout_rate=dropout_rate,
                    training=True,
                    return_hpi=False,
                ).squeeze()

                if y_pred.ndim == 0:
                    y_pred = y_pred.view(1, 1)

                loss_args = (y_pred[: batch.batch_size], quantiles[: batch.batch_size])
                if do_mixup:
                    y_a, y_b = y_a[: batch.batch_size], y_b[: batch.batch_size]
                    lam = lam.squeeze()
                    if lam.ndim == 0:
                        lam = lam.view(1, 1)
                    lam = lam[: batch.batch_size]
                    loss = lam * loss_fn(y_a, *loss_args) + (1 - lam) * loss_fn(y_b, *loss_args)
                    # loss = loss_fn(lam * y_a + (1 - lam) * y_b, *loss_args)
                else:
                    y = y[: batch.batch_size]
                    loss = loss_fn(y, *loss_args)
                loss = loss.mean()
                losses.append(loss.detach().cpu().numpy())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if verbose:
                (
                    f"Epoch {epoch}/{num_epochs}"
                    f" - {int(time.time() - start):d}s"
                    f" - loss: {np.array(losses).mean():.5f}"
                )

            save_name = SAVE_FORMAT.format(model_num=model_num + 1, epoch=epoch)
            save_file = os.path.join(save_path, save_name)
            torch.save(self.state_dict(), save_file)

    def predict(
        self,
        X_dict: dict[str, torch.Tensor],
        _: Any = None,
        quantiles: float | list[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict with the model.

        Args:
            X_dict (dict[str, torch.Tensor]): Input dictionary.
            _ (Any, optional): Hook for arguments to custom methods.
                Defaults to None.
            quantiles (float | list[float] | None, optional):
                Quantiles to predict.
                If None, defaults to [0.25, 0.5, 0.75].

        Returns:
            tuple[np.ndarray, np.ndarray]: Predicted values and indices.

        """
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75]
        elif not isinstance(quantiles, list):
            quantiles = [quantiles]

        y_qs = []
        y_hpi_qs = []
        with torch.no_grad():
            self.eval()
            for q in torch.Tensor(quantiles).to(get_device()):
                y_q, y_q_hpi = self.forward(X_dict, _, q, training=False, return_hpi=True)
                y_qs.append(y_q.squeeze().cpu().data.numpy())
                y_hpi_qs.append(y_q_hpi.squeeze().cpu().data.numpy())
        y = np.stack(y_qs, axis=-1)
        y_hpi = np.stack(y_hpi_qs, axis=-1)
        return y, y_hpi


class NeuralAVM:
    """Neural network AVM for estimating property valuations and indices."""

    def __init__(
        self,
        num_models: int,
        init_dict: dict[str, np.ndarray],
        feature_dict: dict[str, list[str]],
        hidden_dims: list[int] | None = None,
        emb_size: int = 5,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        random_seed: int = 0,
        verbose: bool = True,
    ) -> None:
        """Initialize the model.

        Args:
            num_models (int): Number of models in the ensemble.
            init_dict (dict[str, np.ndarray]): Initialization dictionary.
            feature_dict (dict[str, list[str]]): Feature dictionary.
            hidden_dims (list[int] | None, optional): Hidden layer sizes.
                If None, defaults to [128, 32].
            emb_size (int, optional): Output size for each embedding.
                Defaults to 5.
            dropout_rate (float, optional): Dropout rate for training.
                Defaults to 0.1.
            learning_rate (float): Learning rate for training.
                Defaults to 1e-3.
            random_seed (int, optional): Random seed to use.
                Defaults to 0.
            verbose (bool, optional): Verbose output.
                Defaults to True.

        """
        if hidden_dims is None:
            hidden_dims = [128, 32]
        elif not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]

        self.feature_dict = feature_dict
        self.preprocess_fn = partial(prepare_dataframe, feature_dict=self.feature_dict)

        self._is_fitted = False

        self.models = []
        for i in range(num_models):
            random.seed(i + random_seed)
            np.random.seed(i + random_seed)
            torch.manual_seed(i + random_seed)

            model = BaseNeuralAVM(
                init_dict,
                feature_dict,
                hidden_dims,
                emb_size,
                dropout_rate,
                learning_rate,
            )
            model.to(get_device())

            if verbose and i == 0:
                model_params = filter(lambda p: p.requires_grad, model.parameters())
                sum([np.prod(p.size()) for p in model_params])

            self.models.append(model)

    def prepare_model_input(
        self,
        X: pd.DataFrame | pd.Series | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Prepare model input.

        Converts the input into a preprocessed dictionary of Torch Tensors.

        Args:
            X (pd.DataFrame | pd.Series | torch.Tensor): Input data.

        Returns:
            dict[str, torch.Tensor]: Output dictionary.

        """
        return {k: torch.as_tensor(v) for k, v in self.preprocess_fn(X).items()}

    def _create_train_dataloader(
        self,
        X_train: pd.DataFrame | pd.Series,
        y_train: pd.Series,
        batch_size: int,
    ) -> DataLoader:
        """Create a training data loader.

        Args:
            X_train (pd.DataFrame | pd.Series): Training feature data.
            y_train (pd.Series): Training response data.
            batch_size (int): Training batch size.

        Returns:
            DataLoader: Training data loader.

        """
        train_dataset = TabularDataset(X_train, y_train)
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=partial(
                collate_fn,
                preprocess_fn=self.preprocess_fn,
                columns=list(X_train.columns),
            ),
            worker_init_fn=lambda id: np.random.seed(id),
        )

    def _create_predict_dataloader(self, _: Any) -> None:
        """Create a prediction data loader. Hook for custom function."""
        return

    def _create_explain_dataloader(self, _: Any) -> None:
        """Create an explanation data loader. Hook for custom function."""
        return

    def fit(
        self,
        X_train: pd.DataFrame | pd.Series,
        y_train: pd.Series,
        num_epochs: int,
        batch_size: int,
        save_path: str = SAVE_PATH,
        verbose: bool = True,
        dataloader_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the model.

        Args:
            X_train (pd.DataFrame | pd.Series): Training feature data.
            y_train (pd.Series): Training response data.
            num_epochs (int): Number of epochs to train.
            batch_size (int): Batch size for training.
            save_path (str, optional):
                Defaults to SAVE_PATH.
            verbose (bool, optional): Verbose output.
                Defaults to True.
            dataloader_kwargs (dict | None, optional): Data loader keyword
                arguments.
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Self: Fitted model.

        """
        if os.path.exists(save_path) and os.path.isdir(save_path):
            shutil.rmtree(save_path)

        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        self.num_epochs = num_epochs

        train_dataloader = self._create_train_dataloader(
            X_train,
            y_train,
            batch_size,
            **dataloader_kwargs,
        )

        for model_num, model in enumerate(self.models):
            if verbose:
                pass
            model.fit(
                train_dataloader,
                save_path,
                num_epochs,
                model_num,
                verbose=verbose,
                **kwargs,
            )

        self._is_fitted = True

        return self

    def predict(
        self,
        X_predict: pd.DataFrame | pd.Series,
        quantiles: float | list[float] | None = None,
        min_epoch: int | None = None,
        max_epoch: int | None = None,
        save_path: str = SAVE_PATH,
        dataloader_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict with the model.

        Args:
            X_predict (pd.DataFrame | pd.Series): Prediction data.
            quantiles (float | list[float], optional):
                Quantiles to estimate.
                If None, defaults to [0.25, 0.5, 0.75].
            min_epoch (int | None, optional): Minimum prediction epoch.
                If None, defaults to the final epoch.
            max_epoch (int | None, optional): Maximum prediction epoch.
                If None, defaults to the final epoch.
            save_path (str, optional):
                Defaults to SAVE_PATH.
            dataloader_kwargs (dict | None, optional): Data loader keyword
                arguments.
                Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: Predicted values and indices.

        """
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75]
        elif not isinstance(quantiles, list):
            quantiles = [quantiles]

        if min_epoch is None:
            min_epoch = self.num_epochs
        if max_epoch is None:
            max_epoch = self.num_epochs

        if min_epoch > max_epoch:
            logging.warning(
                f"`min_epoch` {min_epoch} is greater than `max_epoch` {max_epoch}. "
                f"Setting `min_epoch` to {max_epoch}.",
            )
            min_epoch = max_epoch

        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if not self._is_fitted:
            msg = (
                "The model is not fitted yet. Call `fit` with appropriate "
                "arguments before using this model."
            )
            raise NotFittedError(msg)

        X_dict = self.prepare_model_input(X_predict)

        _ = self._create_predict_dataloader(X_predict, **dataloader_kwargs)  # type: ignore

        preds = []
        preds_hpi = []
        for model_num, model in enumerate(self.models):
            for epoch in range(min_epoch, max_epoch + 1):
                save_name = SAVE_FORMAT.format(model_num=model_num + 1, epoch=epoch)
                save_file = os.path.join(save_path, save_name)
                model.load_state_dict(torch.load(save_file))
                y_pred, y_pred_hpi = model.predict(X_dict, _, quantiles=quantiles)
                preds.append(y_pred)
                preds_hpi.append(y_pred_hpi)
        y_pred = np.stack(preds, axis=2).mean(axis=2)
        y_pred_hpi = np.stack(preds_hpi, axis=2).mean(axis=2)
        return y_pred, y_pred_hpi

    def explain(
        self,
        X_explain: pd.DataFrame | pd.Series,
        X_baseline: pd.DataFrame | pd.Series,
        quantile: float | None = None,
        min_epoch: int | None = None,
        max_epoch: int | None = None,
        save_path: str = SAVE_PATH,
        dataloader_kwargs: dict | None = None,
    ) -> pd.DataFrame:
        """Generate explainable feature attributions for input data.

        Args:
            X_explain (pd.DataFrame | pd.Series): Data to explain.
            X_baseline (pd.DataFrame | pd.Series): Baseline data.
            quantile (float | None, optional): Quantile to estimate.
                If None, defaults to 0.5.
            min_epoch (int | None, optional): Minimum prediction epoch.
                If None, defaults to the final epoch.
            max_epoch (int | None, optional): Maximum prediction epoch.
                If None, defaults to the final epoch.
            save_path (str, optional):
                Defaults to SAVE_PATH.
            dataloader_kwargs (dict | None, optional): Data loader keyword
                arguments.
                Defaults to None.

        Returns:
            pd.DataFrame: Feature attributions for each row in the data.

        """
        if quantile is None:
            quantile = 0.5

        if min_epoch is None:
            min_epoch = self.num_epochs
        if max_epoch is None:
            max_epoch = self.num_epochs

        if min_epoch > max_epoch:
            logging.warning(
                f"`min_epoch` {min_epoch} is greater than `max_epoch` {max_epoch}. "
                f"Setting `min_epoch` to {max_epoch}.",
            )
            min_epoch = max_epoch

        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if len(X_baseline) == 0:
            msg = "Must provide at least 1 baseline sample."
            raise ValueError(msg)

        if not self._is_fitted:
            msg = (
                "The model is not fitted yet. Call `fit` with appropriate "
                "arguments before using this model."
            )
            raise NotFittedError(msg)

        num_background = len(X_baseline)
        explainer = DeepLift if num_background == 1 else DeepLiftShap

        X_explain_dict = self.prepare_model_input(X_explain)
        X_baseline_dict = self.prepare_model_input(X_baseline)

        _ = self._create_explain_dataloader(X_explain, **dataloader_kwargs)  # type: ignore

        attrs = []
        for model_num, model in enumerate(self.models):
            input_args = (model.dropout_rate, False)
            additional_forward_args = (_, None, *input_args, False)

            # Determine the input size of each feature for the model.
            f_size_dict = {}
            for f in X_explain_dict:
                f_data = model._forward_inputs({f: X_explain_dict[f][:1]}, *input_args)
                f_size_dict[f] = f_data.detach().cpu().numpy().shape[-1]
            f_inputs = {
                k: ([f"{k}_{v_i}" for v_i in range(v)] if v > 1 else [k])
                for k, v in f_size_dict.items()
            }
            f_inputs["_quantile"] = ["_quantile"]

            for epoch in range(min_epoch, max_epoch + 1):
                save_name = SAVE_FORMAT.format(model_num=model_num + 1, epoch=epoch)
                save_file = os.path.join(save_path, save_name)
                model.load_state_dict(torch.load(save_file))

                inputs = model._forward_inputs(X_explain_dict, *input_args)
                baseline = model._forward_inputs(X_baseline_dict, *input_args)

                # Insert quantile into feature vectors so it can be attributed.
                inputs_q = torch.full((inputs.size(0), 1), quantile).to(inputs.device)
                baseline_q = torch.full((baseline.size(0), 1), 0.5).to(baseline.device)
                inputs = torch.cat([inputs, inputs_q], 1)
                baseline = torch.cat([baseline, baseline_q], 1)

                dl = explainer(model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    attrs_i = dl.attribute(
                        inputs[-len(X_explain) :, :],
                        baseline[-len(X_baseline) :, :],
                        target=0,
                        additional_forward_args=additional_forward_args,
                        return_convergence_delta=False,
                    )

                # Assign feature attributions to a corresponding feature label.
                # If the feature is multidimensional, sum its attributions.
                f_attrs_i = {}
                n_inputs = 0
                for f in f_inputs:
                    f_inputs_size = len(f_inputs[f])
                    f_attr = attrs_i[:, n_inputs : n_inputs + f_inputs_size]
                    f_attrs_i[f] = f_attr.sum(axis=1).tolist()
                    n_inputs += f_inputs_size
                attrs.append(pd.DataFrame(f_attrs_i))

        df_attrs = pd.concat(attrs)
        return df_attrs.groupby(df_attrs.index).mean()
