"""Data utilities."""

import numpy as np
import pandas as pd
import torch


def prepare_dataframe(
    X: pd.DataFrame,
    feature_dict: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Prepare dataframe feature data.

    Transforms a DataFrame of features into a dictionary of NumPy feature
    arrays of the appropriate types.

    Args:
        df (pd.DataFrame): Input feature data.
        feature_dict (dict[str, list[str]]): Input feature dictionary.

    Returns:
        dict[str, np.ndarray]: Dictionary of feature names mapped to NumPy
            arrays.

    """
    X_out = {}

    for key in feature_dict["nulls"]:
        X_out[key] = np.array(X[[key]]).astype(np.float32)

    for key in feature_dict["numerics"] + feature_dict["log_numerics"]:
        X_out[key] = np.array(X[[key]]).astype(np.float32)

    for key in feature_dict["categoricals"] + feature_dict["ordinals"]:
        X_out[key] = np.array(X[[key]]).astype(np.int32)

    return X_out


def prepare_tensor(X: torch.Tensor, feature_dict: dict[str, list[str]]) -> dict[str, torch.Tensor]:
    """Prepare tensor feature data.

    Transforms a Tensor of features into a dictionary of feature tensors of
    the appropriate types.

    Args:
        X (torch.Tensor): Input feature data.
        feature_dict (dict[str, list[str]]): Input feature dictionary.

    Returns:
        dict[str, torch.Tensor]: Dictionary of feature names mapped to
            Tensors.

    """
    X_out = {}

    f_idx = 0
    for d in feature_dict:
        for key in feature_dict[d]:
            if (
                key in feature_dict["nulls"]
                or key in feature_dict["numerics"]
                or key in feature_dict["log_numerics"]
                or key in feature_dict["categoricals"]
                or key in feature_dict["ordinals"]
            ):
                X_out[key] = X[:, f_idx]
            else:
                continue
            f_idx += 1

    return X_out
