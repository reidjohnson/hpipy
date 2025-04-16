from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from hpipy.extensions import NeuralNetworkIndex
from hpipy.extensions.neural.neural_avm.model_layers import MonotonicDense, OrdinalEmbedding
from hpipy.extensions.neural.neural_avm.utils.data import prepare_dataframe, prepare_tensor
from hpipy.extensions.neural.neural_avm.utils.model import mixup, quantile_loss


@pytest.mark.usefixtures("seattle_dataset")
@pytest.mark.parametrize("estimator", ["residual", "attributional"])
def test_nn_create_trans(seattle_dataset: pd.DataFrame, estimator: str) -> None:
    nn_index = NeuralNetworkIndex().create_index(
        seattle_dataset,
        date="sale_date",
        price="sale_price",
        trans_id="sale_id",
        prop_id="pinx",
        dep_var="price",
        ind_var=["area", "baths", "beds", "latitude", "longitude", "tot_sf"],
        estimator=estimator,
        feature_dict={
            "numerics": ["latitude", "longitude"],
            "log_numerics": ["tot_sf"],
            "categoricals": ["area", "sale_date"],
            "ordinals": ["baths", "beds"],
            "hpi": ["sale_date"],
        },
        num_models=1,
        num_epochs=3,
        min_pred_epoch=1,
        verbose=True,
    )
    assert nn_index.model.params["estimator"] == estimator


@pytest.fixture
def device() -> torch.device:
    """Return default device for testing."""
    return torch.device("cpu")


class TestMonotonicDense:
    """Test suite for MonotonicDense layer."""

    @pytest.fixture
    def layer_params(self) -> dict[str, Any]:
        """Return default layer parameters for testing."""
        return {
            "in_features": 3,
            "out_features": 6,
            "activation": nn.ReLU(),
            "monotonicity_indicator": [1, -1, 0],
            "activation_weights": [0.4, 0.4, 0.2],
        }

    def test_initialization(self, layer_params: dict[str, Any], device: torch.device) -> None:
        """Test layer initialization with various parameters."""
        layer = MonotonicDense(**layer_params, device=device)
        assert layer.in_features == layer_params["in_features"]
        assert layer.out_features == layer_params["out_features"]
        assert isinstance(layer.activation, nn.ReLU)

        if layer.monotonicity_indicator is not None:
            assert torch.allclose(
                layer.monotonicity_indicator,
                torch.tensor(layer_params["monotonicity_indicator"], device=device),
            )

        # Test activation weights normalization.
        expected_weights = torch.tensor(layer_params["activation_weights"], device=device)
        expected_weights = expected_weights / expected_weights.sum()
        assert torch.allclose(layer.activation_weights, expected_weights)

    def test_invalid_activation_weights(
        self,
        layer_params: dict[str, Any],
        device: torch.device,
    ) -> None:
        """Test that invalid activation weights raise appropriate errors."""
        with pytest.raises(ValueError):
            invalid_params = layer_params.copy()
            invalid_params["activation_weights"] = [1, 1]  # wrong length
            MonotonicDense(**invalid_params, device=device)

        with pytest.raises(ValueError):
            invalid_params = layer_params.copy()
            invalid_params["activation_weights"] = [-1, 1, 1]  # negative values
            MonotonicDense(**invalid_params, device=device)

    def test_forward_pass(self, layer_params: dict[str, Any], device: torch.device) -> None:
        """Test forward pass with different monotonicity settings."""
        layer = MonotonicDense(**layer_params, device=device)
        x = torch.randn(4, layer_params["in_features"], device=device)
        output = layer(x)

        assert output.shape == (4, layer_params["out_features"])
        assert not torch.isnan(output).any()

    def test_monotonicity_constraints(self, device: torch.device) -> None:
        """Test that monotonicity constraints are properly enforced."""
        # Test increasing monotonicity.
        layer = MonotonicDense(
            in_features=1, out_features=1, monotonicity_indicator=1, device=device
        )
        x1 = torch.tensor([[1.0]], device=device)
        x2 = torch.tensor([[2.0]], device=device)
        y1 = layer(x1)
        y2 = layer(x2)
        assert (y2 >= y1).all()

        # Test decreasing monotonicity.
        layer = MonotonicDense(
            in_features=1, out_features=1, monotonicity_indicator=-1, device=device
        )
        y1 = layer(x1)
        y2 = layer(x2)
        assert (y2 <= y1).all()


class TestOrdinalEmbedding:
    """Test suite for OrdinalEmbedding layer."""

    @pytest.fixture
    def layer_params(self) -> dict[str, Any]:
        """Return default layer parameters for testing."""
        return {
            "num_embeddings": 5,
            "embedding_dim": 3,
        }

    def test_initialization(self, layer_params: dict[str, Any], device: torch.device) -> None:
        """Test layer initialization."""
        layer = OrdinalEmbedding(**layer_params, device=device)
        assert layer.num_embeddings == layer_params["num_embeddings"]
        assert layer.embedding_dim == layer_params["embedding_dim"]
        assert layer.weight.shape == (
            layer_params["embedding_dim"],
            layer_params["num_embeddings"],
        )
        assert layer.null_weight.shape == (layer_params["embedding_dim"],)

    def test_forward_pass(self, layer_params: dict[str, Any], device: torch.device) -> None:
        """Test forward pass with various inputs."""
        layer = OrdinalEmbedding(**layer_params, device=device)

        # Test regular values.
        x = torch.tensor([1, 3, 2, 0, 4], device=device)
        output = layer(x)
        assert output.shape == (5, layer_params["embedding_dim"])
        assert not torch.isnan(output).any()

        # Test null values (zeros).
        x_null = torch.zeros(3, device=device)
        output_null = layer(x_null)
        assert output_null.shape == (3, layer_params["embedding_dim"])
        assert torch.allclose(output_null[0], output_null[1])

    def test_ordinal_encoding(self, device: torch.device) -> None:
        """Test that ordinal encoding is correct."""
        x = torch.tensor([1, 3, 2, 0, 4], device=device)

        expected_encoding = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            device=device,
        )

        with torch.no_grad():
            x = x.reshape(-1, 1)
            actual_encoding = (x > torch.arange(4, device=device)).float()

        assert torch.allclose(actual_encoding, expected_encoding.float())

    def test_parameter_shapes(self, layer_params: dict[str, Any], device: torch.device) -> None:
        """Test that parameter shapes are correct after initialization."""
        layer = OrdinalEmbedding(**layer_params, device=device)

        assert layer.weight.shape == (
            layer_params["embedding_dim"],
            layer_params["num_embeddings"],
        )
        assert layer.null_weight.shape == (layer_params["embedding_dim"],)
        assert layer.bias is None


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("feature_dim", [2, 10])
def test_mixup_shape(batch_size: int, feature_dim: int) -> None:
    """Test mixup output shapes."""
    X = torch.randn(batch_size, feature_dim)
    y = torch.randn(batch_size)

    mixed_X, y_a, y_b, lam = mixup(X, y, alpha=1.0)

    assert mixed_X.shape == X.shape
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape
    assert lam.shape == (batch_size, 1)


def test_mixup_lambda_range() -> None:
    """Test mixup lambda values are in [0, 1]."""
    batch_size, feature_dim = 16, 4
    X = torch.randn(batch_size, feature_dim)
    y = torch.randn(batch_size)

    _, _, _, lam = mixup(X, y, alpha=1.0)

    assert torch.all(lam >= 0) and torch.all(lam <= 1)


def test_mixup_deterministic() -> None:
    """Test mixup with fixed seed for reproducibility."""
    batch_size, feature_dim = 4, 2
    X = torch.randn(batch_size, feature_dim)
    y = torch.randn(batch_size)

    torch.manual_seed(0)
    result1 = mixup(X.clone(), y.clone(), alpha=1.0)

    torch.manual_seed(0)
    result2 = mixup(X.clone(), y.clone(), alpha=1.0)

    for r1, r2 in zip(result1, result2):
        assert torch.allclose(r1, r2)


@pytest.mark.parametrize("batch_size", [4, 8])
def test_quantile_loss_shape(batch_size: int) -> None:
    """Test quantile loss output shape."""
    y_true = torch.randn(batch_size)
    y_pred = torch.randn(batch_size)
    quantile = torch.tensor(0.5)

    loss = quantile_loss(y_true, y_pred, quantile)
    assert loss.shape == y_true.shape


@pytest.mark.parametrize("quantile", [0.1, 0.5, 0.9])
def test_quantile_loss_values(quantile: float) -> None:
    """Test quantile loss computation for different quantiles."""
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([1.5, 1.5, 1.5])
    q = torch.tensor(quantile)

    loss = quantile_loss(y_true, y_pred, q)

    assert torch.all(loss >= 0)


def test_quantile_loss_zero() -> None:
    """Test quantile loss is zero when prediction equals true value."""
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    quantile = torch.tensor(0.5)

    loss = quantile_loss(y_true, y_pred, quantile)
    assert torch.all(loss == 0)


def test_prepare_dataframe() -> None:
    """Test prepare_dataframe function."""
    test_data = pd.DataFrame(
        {
            "null_feature": [1.0, np.nan, 3.0],
            "numeric_feature": [1.5, 2.5, 3.5],
            "log_numeric_feature": [10.0, 20.0, 30.0],
            "categorical_feature": [0, 1, 2],
            "ordinal_feature": [1, 2, 3],
        }
    )

    feature_dict = {
        "nulls": ["null_feature"],
        "numerics": ["numeric_feature"],
        "log_numerics": ["log_numeric_feature"],
        "categoricals": ["categorical_feature"],
        "ordinals": ["ordinal_feature"],
    }

    result = prepare_dataframe(test_data, feature_dict)

    assert isinstance(result, dict)
    assert len(result) == 5

    # Check data types.
    assert result["null_feature"].dtype == np.float32
    assert result["numeric_feature"].dtype == np.float32
    assert result["log_numeric_feature"].dtype == np.float32
    assert result["categorical_feature"].dtype == np.int32
    assert result["ordinal_feature"].dtype == np.int32

    # Check values.
    np.testing.assert_array_equal(
        result["numeric_feature"], np.array([[1.5], [2.5], [3.5]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        result["categorical_feature"], np.array([[0], [1], [2]], dtype=np.int32)
    )


def test_prepare_tensor() -> None:
    """Test prepare_tensor function."""
    test_tensor = torch.tensor(
        [
            [1.0, 1.5, 10.0, 0.0, 1.0],
            [float("nan"), 2.5, 20.0, 1.0, 2.0],
            [3.0, 3.5, 30.0, 2.0, 3.0],
        ],
        dtype=torch.float32,
    )

    feature_dict = {
        "nulls": ["null_feature"],
        "numerics": ["numeric_feature"],
        "log_numerics": ["log_numeric_feature"],
        "categoricals": ["categorical_feature"],
        "ordinals": ["ordinal_feature"],
    }

    result = prepare_tensor(test_tensor, feature_dict)

    assert isinstance(result, dict)
    assert len(result) == 5

    # Check that all outputs are tensors.
    for key in result:
        assert isinstance(result[key], torch.Tensor)

    # Check values for a few features.
    assert torch.equal(result["numeric_feature"], test_tensor[:, 1])
    assert torch.equal(result["categorical_feature"], test_tensor[:, 3])
