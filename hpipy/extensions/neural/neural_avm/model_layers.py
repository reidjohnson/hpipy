"""Custom model layers."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MonotonicDense(nn.Linear):
    """Monotonic counterpart of the regular dense (linear) layer.

    Implements a monotonically constrained layer as described in [1].

    References:
        [1] Runje, Davor, and Sharath M. Shankaranarayana. "Constrained
            Monotonic Neural Networks." International Conference on Machine
            Learning. PMLR, 2023.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module | None = None,
        monotonicity_indicator: int | list[int] | None = 1,
        activation_weights: list[int] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            activation (Optional[nn.Module], optional): Activation function.
                Defaults to None.
            monotonicity_indicator (int | list[int] | None, optional):
                Vector to indicate which of the inputs are monotonically
                increasing, monotonically decreasing, or non-monotonic. Has
                value 1 for monotonically increasing, -1 for monotonically
                decreasing, and 0 for non-monotonic variables.
                Defaults to 1.
            activation_weights (list[int] | None = None, optional): Activation
                weights for convex, concave, and saturated units, respectively.
                Defaults to None. If None, defaults to [1, 1, 1].
            device (torch.device | None, optional): Device on which to
                initialize the layer. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the layer.
                Defaults to None.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, **factory_kwargs, **kwargs)  # type: ignore

        self.activation = activation

        if monotonicity_indicator is not None:
            self.register_buffer(
                "monotonicity_indicator",
                torch.tensor(monotonicity_indicator, **factory_kwargs),  # type: ignore
            )
        else:
            self.monotonicity_indicator = None

        if activation_weights is None:
            activation_weights = [1, 1, 1]
        else:
            if len(activation_weights) != 3:
                msg = (
                    "`activation_weights` must have exactly three components, got "
                    f"{activation_weights}."
                )
                raise ValueError(
                    msg,
                )
            if (np.array(activation_weights) < 0).any():
                msg = (
                    "`activation_weights` values must be non-negative, got "
                    f"{activation_weights}."
                )
                raise ValueError(
                    msg,
                )
        self.activation_weights = torch.Tensor(activation_weights, device=device)
        self.activation_weights = self.activation_weights / self.activation_weights.sum(dim=0)

        self.n_convex = round(self.activation_weights[0].item() * self.out_features)
        self.n_concave = round(self.activation_weights[1].item() * self.out_features)
        self.n_saturated = self.out_features - self.n_convex - self.n_concave

    @contextmanager
    def _replace_kernel(self) -> Generator:
        """Replace kernel weights with signed values as indicated."""
        original_weight = self.weight

        if self.monotonicity_indicator is not None:
            weight_abs = torch.abs(self.weight)

            if self.monotonicity_indicator.dim == 0:
                if torch.all(self.monotonicity_indicator == 1):
                    self.weight.data = weight_abs
                elif torch.all(self.monotonicity_indicator == -1):
                    self.weight.data = -weight_abs
            else:
                w_ = torch.where(self.monotonicity_indicator == 1, weight_abs, self.weight)
                self.weight.data = torch.where(self.monotonicity_indicator == -1, -weight_abs, w_)
        yield

        self.weight = original_weight

    def _activation_selector(self, x: torch.Tensor) -> torch.Tensor:
        """Select activation sign."""
        return torch.cat(
            [
                torch.ones((x.shape[0], self.n_convex), dtype=torch.int, device=x.device),
                -torch.ones((x.shape[0], self.n_concave), dtype=torch.int, device=x.device),
                torch.zeros((x.shape[0], self.n_saturated), dtype=torch.int, device=x.device),
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            Output data.

        """
        with self._replace_kernel():
            y = super().forward(x)

        if self.activation is not None:
            activation_selector = self._activation_selector(x)

            c = self.activation(torch.ones_like(y))

            y1 = self.activation(y)
            y2 = -self.activation(-y)
            y3 = torch.where(y <= 0, self.activation(y + 1) - c, -self.activation(-(y - 1)) + c)

            y = torch.where(activation_selector == 1, y1, y)
            y = torch.where(activation_selector == -1, y2, y)
            y = torch.where(activation_selector == 0, y3, y)

        return y


class OrdinalEmbedding(nn.Module):
    """Ordinal embedding layer.

    Embeds an ordinal variable using a cumulative or "binary counting"
    encoding, where each ordinal value is encoded as an cumulative binary
    array. Null values are assumed to be the value zero; they are represented
    by an encoding consisting of only zeros and are handled separately by the
    embedding layer from non-null encodings.

    For example, the values [1, 3, 2, 0, 4] are encoded as:

        [[1, 0, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0],
         [1, 1, 1, 1]]

    The embedding applies a single linear layer to all non-zero encodings and
    a separate set of parameters of the same output size to the zero encoding.
    The outputs are additively combined to produce the final embedding output.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the layer.

        Args:
            num_embeddings (int): Size of the dictionary of embeddings.
            embedding_dim (int): Size of each embedding vector.
            device (torch.device | None, optional): Device on which to
                initialize the layer. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the layer.
                Defaults to None.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(
                (self.embedding_dim, self.num_embeddings),
                **factory_kwargs,  # type: ignore
            ),
        )
        self.bias: torch.Tensor | None
        self.register_parameter("bias", None)
        self.null_weight = nn.Parameter(
            torch.empty(self.embedding_dim, **factory_kwargs),  # type: ignore
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the layer parameters."""
        std = 0.01  # small value given the additive effects of the encoding
        nn.init.normal_(self.weight, std=std)
        nn.init.normal_(self.null_weight, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            Output data.

        """
        x = x.reshape(-1, 1)
        nonnull_mask = x.any(dim=1)
        x = (x > torch.arange(self.num_embeddings, device=x.device)).float().to(x.device)
        x_out = torch.zeros(x.size(0), self.embedding_dim, device=x.device)
        x_out[nonnull_mask] += F.linear(x[nonnull_mask], self.weight, self.bias)
        x_out[~nonnull_mask] += self.null_weight
        return x_out

    def extra_repr(self) -> str:
        """Format display representation.

        Returns:
            String representation.

        """
        s = "{num_embeddings}, {embedding_dim}"
        return s.format(**self.__dict__)
