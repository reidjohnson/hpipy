"""Data loader functions."""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Batch(NamedTuple):
    """Batch object."""

    x: dict[str, torch.Tensor]
    y: torch.Tensor
    batch_size: int


def collate_fn(
    batch: list[tuple[np.ndarray, np.ndarray]],
    preprocess_fn: Callable[[pd.DataFrame], dict[str, np.ndarray]],
    columns: list[str],
) -> Batch:
    """Collate a list of batch inputs.

    Feature data is transformed from a list of NumPy arrays into a single
    dictionary of batch-length tensors keyed by the feature column names.

    Response data is transformed from a list of NumPy arrays into a tensor.

    Args:
        batch (list[tuple[np.ndarray, np.ndarray]]): Inputs.
        preprocess_fn (Callable[[pd.DataFrame], dict[str, np.ndarray]]):
            Preprocessing function.
        columns (list[str]): List of column names.

    Returns:
        X as dict[str, torch.Tensor], y as torch.Tensor.

    """
    x, y = zip(*batch, strict=False)
    x = preprocess_fn(pd.DataFrame(np.stack(x), columns=columns))
    x = {k: torch.from_numpy(v) for k, v in x.items()}
    y = torch.from_numpy(np.stack(y))
    return Batch(x=x, y=y, batch_size=len(y))


class TabularDataset(Dataset):
    """Tabular dataset implementation for PyTorch."""

    def __init__(
        self,
        X: pd.DataFrame | pd.Series | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ) -> None:
        """Initialize the tabular dataset.

        Args:
            X (pd.DataFrame | pd.Series | np.ndarray): Input feature data.
            y (pd.Series | np.ndarray | None, optional): Input
                response data.
                Defaults to None.

        """
        self.X = X
        self.y = y
        if isinstance(self.X, (pd.DataFrame | pd.Series)):
            self.X = self.X.to_numpy()
        if isinstance(self.y, pd.Series):
            self.y = self.y.to_numpy()

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Return indexed X and y data."""
        X = self.X[idx]
        if self.y is not None:
            y = self.y[idx]
            return X, y
        return X

    def __len__(self) -> int:
        """Return length of data."""
        return len(self.X)
