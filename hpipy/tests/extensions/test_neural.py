import pandas as pd
import pytest

from hpipy.extensions import GraphNeuralNetworkIndex, NeuralNetworkIndex


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

    gnn_index = GraphNeuralNetworkIndex().create_index(
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
    assert gnn_index.model.params["estimator"] == estimator
