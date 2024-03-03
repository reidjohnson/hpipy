"""Graph neural network (GNN) AVM for jointly estimating property valuations
and HPIs.

The module structure is the following:

- The ``BaseGraphNeuralAVM`` class extends the ``BaseNeuralAVM`` class to
  implement a ``forward`` method for training and predicting with an
  individual graph neural network.

- The ``GraphNeuralAVM`` class extends the ``NeuralAVM`` class to implement
  transaction graph construction and corresponding graph loader methods.
"""


import copy
import datetime
import math
import random
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import TransformerConv

from .model_fnn import BaseNeuralAVM, NeuralAVM
from .utils.data import prepare_dataframe, prepare_tensor
from .utils.model import get_device

CURRENT_TIME = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

SAVE_PATH = "../model_outputs/neural_avm/" + CURRENT_TIME
SAVE_FORMAT = "model_{model_num:03d}_epoch_{epoch:03d}.pt"


class BaseGraphNeuralAVM(BaseNeuralAVM):
    """Base graph neural network (GNN) AVM."""

    def __init__(
        self,
        init_dict: dict[str, np.ndarray],
        feature_dict: dict[str, list[str]],
        conv_dims: list[int],
        dense_dims: list[int],
        emb_size: int,
        edge_feat_dim: int,
        mha_heads: int,
        dropout_rate: float,
        learning_rate: float,
    ) -> None:
        """Initialize the model.

        Args:
            init_dict (dict[str, np.ndarray]): Initialization dictionary.
            feature_dict (dict[str, list[str]]): Feature dictionary.
            conv_dims (list[int]): Convolutional layer sizes.
            dense_dims (list[int]): Dense layer sizes.
            emb_size (int): Output size for each embedding.
            edge_feat_dim (int): Edge features size.
            mha_heads (int): Number of multi-head attention heads.
            dropout_rate (float): Dropout rate for training.
            learning_rate (float): Learning rate for training.
        """
        nn.Module.__init__(self)

        input_size_prp, input_size_hpi, x_emb = self._define_inputs(
            init_dict,
            feature_dict,
            emb_size,
        )

        convs_prp, convs_hpi = self._define_convs(
            input_size_prp,
            input_size_hpi,
            conv_dims,
            edge_feat_dim,
            mha_heads,
            dropout_rate,
        )
        layers_prp, layers_hpi = self._define_dense(
            conv_dims[-1],
            conv_dims[-1] if convs_hpi else 0,
            dense_dims,
            dropout_rate=0,
        )

        self.x_emb = nn.ModuleDict(x_emb)
        self.convs_prp = nn.ModuleList(convs_prp)
        self.convs_hpi = nn.ModuleList(convs_hpi)
        self.dense_prp = nn.Sequential(*layers_prp)
        self.dense_hpi = nn.Sequential(*layers_hpi)

        params = [
            {"params": self.x_emb.parameters()},
            {"params": self.convs_prp.parameters()},
            {"params": self.convs_hpi.parameters()},
            {"params": self.dense_prp.parameters()},
            {"params": self.dense_hpi.parameters()},
        ]
        self.optimizer = self._optimizer(params, learning_rate)

        self.feature_dict = feature_dict
        self.edge_feat_dim = edge_feat_dim
        self.dropout_rate = dropout_rate
        self.input_size_prp = input_size_prp
        self.input_size_hpi = input_size_hpi

    def _define_convs(
        self,
        input_size_prp: int,
        input_size_hpi: int,
        conv_dims: list[int],
        edge_feat_dim: int,
        mha_heads: int,
        dropout_rate: float,
    ) -> Tuple[list[nn.Module], list[nn.Module]]:
        """Define the convolutional layers."""
        # Define the property-level convolutional layers.
        convs_prp: list[nn.Module] = []
        sizes_conv_prp = [input_size_prp] + conv_dims
        for i, (n_in, n_out) in enumerate(zip(sizes_conv_prp[:-1], sizes_conv_prp[1:])):
            if i == 0 and dropout_rate > 0:
                # Use dropout to bias the learning to index-level layers.
                convs_prp.append(nn.Dropout(dropout_rate))
            convs_prp.append(
                TransformerConv(
                    n_in,
                    n_out,
                    edge_dim=None if edge_feat_dim == 0 else edge_feat_dim,
                    heads=mha_heads if i + 2 < len(sizes_conv_prp) else 1,
                    concat=False if i + 2 < len(sizes_conv_prp) else True,
                )
            )

        # Define the index-level convolutional layers.
        convs_hpi: list[nn.Module] = []
        if input_size_hpi > 0:
            sizes_conv_hpi = [input_size_hpi] + conv_dims
            for i, (n_in, n_out) in enumerate(zip(sizes_conv_hpi[:-1], sizes_conv_hpi[1:])):
                convs_hpi.append(
                    TransformerConv(
                        n_in,
                        n_out,
                        edge_dim=None if edge_feat_dim == 0 else edge_feat_dim,
                        heads=mha_heads if i + 2 < len(sizes_conv_prp) else 1,
                        concat=False if i + 2 < len(sizes_conv_prp) else True,
                    )
                )

        return convs_prp, convs_hpi

    def _forward_convs(
        self,
        x_prp: torch.Tensor,
        x_hpi: torch.Tensor,
        subgraph: Union[NeighborLoader, Data],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the convolutional layers."""
        # Compute representations of nodes layer by layer, using all available
        # edges. This leads to faster computation in contrast to immediately
        # computing the final representations of each batch.
        if isinstance(subgraph, NeighborLoader):
            for i, conv in enumerate(self.convs_prp):
                x_prps = []
                for batch in subgraph:
                    edge_attr = None if self.edge_feat_dim == 0 else batch.edge_attr
                    x_prp_i = x_prp[batch.n_id.to(x_prp.device)]
                    if isinstance(conv, nn.Dropout):
                        x_prp_i = conv(x_prp_i)
                    else:
                        x_prp_i = conv(x_prp_i, batch.edge_index, edge_attr)
                        if i < len(self.convs_prp) - 1:
                            x_prp_i = x_prp_i.relu_()
                    x_prps.append(x_prp_i[: batch.batch_size].cpu())
                x_prp = torch.cat(x_prps, dim=0)
            for i, conv in enumerate(self.convs_hpi):
                x_hpis = []
                for batch in subgraph:
                    edge_attr = None if self.edge_feat_dim == 0 else batch.edge_attr
                    x_hpi_i = x_hpi[batch.n_id.to(x_hpi.device)]
                    x_hpi_i = conv(x_hpi_i, batch.edge_index, edge_attr)
                    if i < len(self.convs_hpi) - 1:
                        x_hpi_i = x_hpi_i.relu_()
                    x_hpis.append(x_hpi_i[: batch.batch_size].cpu())
                x_hpi = torch.cat(x_hpis, dim=0)
        elif isinstance(subgraph, Data):
            edge_attr = None if self.edge_feat_dim == 0 else subgraph.edge_attr
            for i, conv in enumerate(self.convs_prp):
                if isinstance(conv, nn.Dropout):
                    x_prp = conv(x_prp)
                else:
                    x_prp = conv(x_prp, subgraph.edge_index, edge_attr)
                    if i < len(self.convs_prp) - 1:
                        x_prp = x_prp.relu_()
            for i, conv in enumerate(self.convs_hpi):
                x_hpi = conv(x_hpi, subgraph.edge_index, edge_attr)
                if i < len(self.convs_hpi) - 1:
                    x_hpi = x_hpi.relu_()
        else:
            raise ValueError
        return x_prp, x_hpi

    def forward(
        self,
        x: Union[dict[str, torch.Tensor], torch.Tensor],
        subgraph: Optional[Union[NeighborLoader, Data]] = None,
        quantiles: Optional[Union[torch.Tensor, list[float]]] = None,
        dropout_rate: Optional[float] = None,
        training: bool = False,
        return_hpi: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward pass.

        Args:
            x (Union[dict[str, torch.Tensor], torch.Tensor]): Inputs.
            subgraph (Optional[Union[NeighborLoader, Data]], optional):
                Transaction subgraph.
                Defaults to None.
            quantiles (Optional[Union[torch.Tensor, list[float]]], optional):
                Quantiles to estimate.
                Defaults to None
            dropout_rate (Optional[float], optional): Dropout rate.
                Defaults to None.
            training (bool, optional): Training Boolean.
                Defaults to False.
            return_hpi (bool, optional): Return HPI component Boolean.
                Defaults to False.

        Returns:
            Estimated property value (and index component).
        """
        if isinstance(x, dict):
            x_in = self._forward_inputs(x, dropout_rate, training)
        elif isinstance(x, torch.Tensor):
            x_in = x
        else:
            raise ValueError(f"{type(x)}")

        # Separate the property and index pathway inputs.
        x_prp = x_in[:, : self.input_size_prp]
        x_hpi = x_in[:, self.input_size_prp : self.input_size_prp + self.input_size_hpi]

        x_prp, x_hpi = self._forward_convs(x_prp, x_hpi, subgraph)
        x_prp, x_hpi = self._forward_dense(x_prp, x_hpi, quantiles)

        x_out = x_prp + x_hpi

        return (x_out, x_hpi) if return_hpi else x_out


class GraphNeuralAVM(NeuralAVM):
    """Graph neural network (GNN) AVM."""

    def __init__(
        self,
        num_models: int,
        init_dict: dict[str, np.ndarray],
        feature_dict: dict[str, list[str]],
        conv_dims: Optional[list[int]] = None,
        dense_dims: Optional[list[int]] = None,
        emb_size: int = 5,
        use_edge_features: bool = False,
        mha_heads: int = 4,
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
            conv_dims (Optional[list[int]], optional): Convolutional layer
                sizes.
                If None, defaults to [32, 32].
            dense_dims (Optional[list[int]], optional): Dense layer sizes.
                If None, defaults to [32].
            emb_size (int, optional): Output size for each embedding.
                Defaults to 5.
            use_edge_features (bool, optional): Use edge features.
                Defaults to False.
            mha_heads (int, optional): Number of multi-head attention heads.
                Defaults to 4.
            dropout_rate (float, optional): Dropout rate for training.
                Defaults to 0.1.
            learning_rate (float): Learning rate for training.
                Defaults to 1e-3.
            random_seed (int, optional): Random seed to use.
                Defaults to 0.
            verbose (bool, optional): Verbose output.
                Defaults to True.
        """
        if conv_dims is None:
            conv_dims = [32, 32]
        elif not isinstance(conv_dims, list):
            conv_dims = [conv_dims]

        if dense_dims is None:
            dense_dims = [32]
        elif not isinstance(dense_dims, list):
            dense_dims = [dense_dims]

        if use_edge_features:
            edge_feat_dim = (len(feature_dict["numerics"] + feature_dict["log_numerics"])) * 2
        else:
            edge_feat_dim = 0

        self.feature_dict = feature_dict

        def _preprocess_fn(
            x: Union[torch.Tensor, pd.DataFrame], feature_dict: dict[str, list[str]]
        ) -> Union[dict[str, np.ndarray], dict[str, torch.Tensor]]:
            if isinstance(x, torch.Tensor):
                return prepare_tensor(x, feature_dict=feature_dict)
            elif isinstance(x, pd.DataFrame):
                return prepare_dataframe(x, feature_dict=feature_dict)
            else:
                raise ValueError

        self.preprocess_fn = partial(
            _preprocess_fn, feature_dict=self.feature_dict  # type: ignore
        )

        self.models = []
        for i in range(num_models):
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

            model = BaseGraphNeuralAVM(
                init_dict,
                feature_dict,
                conv_dims,
                dense_dims,
                emb_size,
                edge_feat_dim,
                mha_heads,
                dropout_rate,
                learning_rate,
            )
            model.to(get_device())

            if verbose and i == 0:
                model_params = filter(lambda p: p.requires_grad, model.parameters())
                params = sum([np.prod(p.size()) for p in model_params])
                print(f"Number of Parameters: {params}")
                print(model)

            self.models.append(model)

    @staticmethod
    def _construct_graph(
        X: pd.DataFrame,
        y: pd.Series,
        feature_dict: dict[str, list[str]],
        sale_date: pd.Series,
        latitude: Optional[pd.Series],
        longitude: Optional[pd.Series],
        k: int = 10,
        training: bool = False,
    ) -> Data:
        """Construct a transaction graph.

        Args:
            X (pd.DataFrame): Input feature data.
            y (pd.Series): Input response data.
            feature_dict (dict[str, list[str]]): Feature dictionary.
            sale_date (Optional[pd.Series], optional): Sale date values for X.
                Defaults to None.
            latitude (Optional[pd.Series], optional): Latitude values for X.
                Defaults to None.
            longitude (Optional[pd.Series], optional): Longitude values for X.
                Defaults to None.
            k (int, optional): Number of neighbors in the graph.
                Defaults to 10.
            training (bool, optional): Training phase.
                Defaults to False.

        Returns:
            Data: Transaction graph.
        """
        if latitude is None:
            raise ValueError
        if longitude is None:
            raise ValueError

        lat = latitude.values
        lng = longitude.values

        feat_cols = [f for type in feature_dict.keys() for f in feature_dict[type]]
        n_num_features = len(feature_dict["numerics"] + feature_dict["log_numerics"])

        # Create the data.
        data_x = torch.Tensor(X[feat_cols].values)
        data_y = torch.Tensor(y.values)
        data = Data(x=data_x, y=data_y)

        idx = torch.arange(len(X))

        if training:
            # Get the training mask.
            train_idx = torch.arange(len(X))
            data.train_mask = torch.isin(idx, train_idx)

        # Get the relative days since beginning for each node.
        min_date = sale_date.min()
        t = (sale_date - min_date).dt.days

        # Get coordinates and set as `data.pos`; used for KNN graph creation.
        lat = lat * (math.pi / 180.0)
        lng = lng * (math.pi / 180.0)
        R = 6371  # approximate Earth radius in km
        x = R * np.cos(lat) * np.cos(lng)
        y = R * np.cos(lat) * np.sin(lng)
        z = R * np.sin(lat)
        data.pos = torch.Tensor(np.stack((x, y, z), axis=1))

        # Get the initial edge index produced by KNN.
        edge_transform = T.KNNGraph(k=k, force_undirected=False)
        data = edge_transform(data)

        # Fix the time leakage; remove all links that go from future to past.
        time_edge = data.edge_index.clone()
        t_dict = dict(zip(idx.numpy(), t.values))
        t_0 = time_edge[0].apply_(t_dict.get)
        t_1 = time_edge[1].apply_(t_dict.get)
        edge_index = data.edge_index.clone()
        new_edge = torch.cat((edge_index, torch.unsqueeze(t_1 - t_0, 0)), 0)
        edge_index = new_edge[:, new_edge[2] > 0][:2]
        data.edge_index = edge_index

        # Add the coordinates and the Euclidean distance as edge features.
        cartesian_transform = T.Cartesian(norm=False)
        euclidean_transform = T.Distance(norm=False)
        data = euclidean_transform(cartesian_transform(data))

        # Add numerical edge feature differences as edge features.
        src, dst = data.edge_index
        new_edge_attr = data.x[:, :n_num_features][dst] - data.x[:, :n_num_features][src]
        edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=1)
        data.edge_attr = edge_attr

        return data

    def _create_train_dataloader(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series,
        batch_size: int,
        k: int = 10,
        num_neighbors: Optional[Union[int, list[int]]] = None,
        sale_date: Optional[pd.Series] = None,
        latitude: Optional[pd.Series] = None,
        longitude: Optional[pd.Series] = None,
    ) -> NeighborLoader:
        """Create a training data loader.

        Args:
            X_train (Union[pd.DataFrame, pd.Series]): Training feature data.
            y_train (pd.Series): Training response data.
            batch_size (int): Training batch size.
            k (int, optional): Number of neighbors in the graph.
                Defaults to 10.
            num_neighbors (Optional[Union[int, list[int]]], optional): Number
                of neighbors in the data loader for each hop of the graph.
                If None, defaults to [5, 5].
            sale_date (Optional[pd.Series], optional): Sale date values for X.
                Defaults to None.
            latitude (Optional[pd.Series], optional): Latitude values for X.
                Defaults to None.
            longitude (Optional[pd.Series], optional): Longitude values for X.
                Defaults to None.

        Returns:
            NeighborLoader: Training data loader.
        """
        if num_neighbors is None:
            num_neighbors = [5, 5]
        elif not isinstance(num_neighbors, list):
            num_neighbors = [num_neighbors]

        train_data = self._construct_graph(
            X_train,
            y_train,
            self.feature_dict,
            sale_date,
            longitude,
            latitude,
            k=k,
            training=True,
        )

        self.graph_data = {
            "X_train": X_train,
            "y_train": y_train,
            "sale_date": sale_date,
            "latitude": latitude,
            "longitude": longitude,
            "train_idx": np.where(train_data.train_mask)[0],
            "edge_feat_dim": train_data.edge_attr.shape[1],
        }

        train_dataloader = NeighborLoader(
            train_data,
            input_nodes=train_data.train_mask,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=True,
        )
        return train_dataloader

    def _create_predict_dataloader(
        self,
        X_predict: Union[pd.DataFrame, pd.Series],
        batch_size: Optional[int] = None,
        k: int = 10,
        sale_date: Optional[pd.Series] = None,
        latitude: Optional[pd.Series] = None,
        longitude: Optional[pd.Series] = None,
    ) -> NeighborLoader:
        """Create a prediction data loader.

        Args:
            X_predict (Union[pd.DataFrame, pd.Series]): Prediction feature
                data.
            batch_size (Optional[int], optional): Prediction batch size.
                If None, defaults to size of the data.
            k (int, optional): Number of neighbors in the graph.
                Defaults to 10.
            sale_date (Optional[pd.Series], optional): Sale date values for X.
                Defaults to None.
            latitude (Optional[pd.Series], optional): Latitude values for X.
                Defaults to None.
            longitude (Optional[pd.Series], optional): Longitude values for X.
                Defaults to None.

        Returns:
            NeighborLoader: Prediction data loader.
        """
        X = pd.concat([self.graph_data["X_train"], X_predict])
        y = self.graph_data["y_train"]

        if batch_size is None:
            batch_size = len(X)

        test_data = self._construct_graph(
            X,
            y,
            self.feature_dict,
            pd.concat([self.graph_data["sale_date"], sale_date]),
            pd.concat([self.graph_data["longitude"], longitude]),
            pd.concat([self.graph_data["latitude"], latitude]),
            k=k,
            training=False,
        )

        subgraph_loader = NeighborLoader(
            copy.copy(test_data),
            input_nodes=None,
            num_neighbors=[-1],
            batch_size=batch_size,
            shuffle=False,
        )

        subgraph_loader.data.num_nodes = test_data.num_nodes
        subgraph_loader.data.n_id = torch.arange(test_data.num_nodes)

        # No need to maintain these features during evaluation.
        del subgraph_loader.data.x, subgraph_loader.data.y

        return subgraph_loader

    def _create_explain_dataloader(
        self,
        X_explain: Union[pd.DataFrame, pd.Series],
        batch_size: Optional[int] = None,
        k: int = 10,
        sale_date: Optional[pd.Series] = None,
        latitude: Optional[pd.Series] = None,
        longitude: Optional[pd.Series] = None,
    ) -> NeighborLoader:
        """Create an explanation data loader.

        Args:
            X_explain (Union[pd.DataFrame, pd.Series]): Explanation feature
                data.
            batch_size (Optional[int], optional): Prediction batch size.
                If None, defaults to size of the data.
            k (int, optional): Number of neighbors in the graph.
                Defaults to 10.
            sale_date (Optional[pd.Series], optional): Sale date values for X.
                Defaults to None.
            latitude (Optional[pd.Series], optional): Latitude values for X.
                Defaults to None.
            longitude (Optional[pd.Series], optional): Longitude values for X.
                Defaults to None.

        Returns:
            NeighborLoader: Explanation data loader.
        """
        X = X_explain.copy()
        X[:] = 0

        # Account for Captum explainer's duplication of input rows.
        # X = pd.DataFrame(np.repeat(X.values, 2, axis=0), columns=X.columns)
        X = pd.concat([X, X]).reset_index(drop=True)

        if batch_size is None:
            batch_size = len(X_explain)

        test_data = self._construct_graph(
            X,
            pd.Series([np.nan]),
            self.feature_dict,
            sale_date,
            longitude,
            latitude,
            k=k,
            training=False,
        )
        test_data.edge_index = torch.Tensor([[], []]).long()

        subgraph_loader = NeighborLoader(
            copy.copy(test_data),
            input_nodes=None,
            num_neighbors=[-1],
            batch_size=batch_size,
            shuffle=False,
        )

        subgraph_loader.data.num_nodes = test_data.num_nodes
        subgraph_loader.data.n_id = torch.arange(test_data.num_nodes)

        # No need to maintain these features during evaluation.
        del subgraph_loader.data.x, subgraph_loader.data.y

        return subgraph_loader

    def prepare_model_input(
        self,
        X: Union[pd.DataFrame, pd.Series, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Prepare model input.

        Converts the input into a preprocessed dictionary of Torch Tensors.

        Args:
            X (Union[pd.DataFrame, pd.Series, torch.Tensor]): Input data.

        Returns:
            dict[str, torch.Tensor]: Output dictionary.
        """
        X = pd.concat([self.graph_data["X_train"], X])
        return {k: torch.as_tensor(v) for k, v in self.preprocess_fn(X).items()}
