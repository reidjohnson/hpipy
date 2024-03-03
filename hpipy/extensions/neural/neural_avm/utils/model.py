"""Model utilities."""

from typing import Optional, Tuple

import torch


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """Get PyTorch device from GPU ID.

    Args:
        gpu_id (Optional[int], optional): GPU ID.
            Defaults to None.

    Returns:
        torch.device: Device based on GPU ID.
    """
    device: str = "cuda:0" if gpu_id is None else (f"cuda:{gpu_id}")
    return torch.device(device if torch.cuda.is_available() else "cpu")


def mixup(
    X: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mixup function for use as a data augmentation method.

    Implements mixup as described in [1], where new training samples are
    created by taking a convex combination of pairwise features and labels.

    Args:
        X (torch.Tensor): Input tensor representing training features.
        y (torch.Tensor): Input tensor representing training labels.
        alpha (float, optional): Beta distribution parameter.
            Defaults to 1.

    Returns:
        The mixed-up inputs, pairs of responses, and lambda.

    References:
        [1] Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David
            Lopez-Paz. "mixup: Beyond Empirical Risk Minimization."
            International Conference on Learning Representations. ICLR, 2018.
    """
    indices = torch.randperm(y.size(0))
    beta = torch.distributions.Beta(alpha, alpha)
    # lam = beta.sample().to(get_device())
    lam = beta.sample(y.reshape(-1, 1).size()).to(get_device())
    X = lam * X + (1 - lam) * X[indices]
    y_a, y_b = y, y[indices]
    return X, y_a, y_b, lam


def quantile_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    quantile: torch.Tensor,
) -> torch.Tensor:
    """Quantile regression or "tilted" loss.

    Args:
        y_true (torch.Tensor): True response values.
        y_pred (torch.Tensor): Predicted response values.
        quantile (torch.Tensor): Quantile to estimate.

    Returns:
        Quantile loss.
    """
    error = y_pred - y_true
    mask = (error.ge(0).float() - quantile).detach()
    return mask * error
