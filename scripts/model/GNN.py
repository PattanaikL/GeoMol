import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_sum


class MLP(nn.Module):
    """
    Creates a NN using nn.ModuleList to automatically adjust the number of layers.
    For each hidden layer, the number of inputs and outputs is constant.

    Inputs:
        in_dim (int):               number of features contained in the input layer.
        out_dim (int):              number of features input and output from each hidden layer, 
                                    including the output layer.
        num_layers (int):           number of layers in the network
        activation (torch function): activation function to be used during the hidden layers
    """
    def __init__(self, in_dim, out_dim, num_layers, activation=torch.nn.ReLU(), layer_norm=False, batch_norm=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        h_dim = in_dim if out_dim < 10 else out_dim

        # create the input layer
        for layer in range(num_layers):
            if layer == 0:
                self.layers.append(nn.Linear(in_dim, h_dim))
            else:
                self.layers.append(nn.Linear(h_dim, h_dim))
            if layer_norm: self.layers.append(nn.LayerNorm(h_dim))
            if batch_norm: self.layers.append(nn.BatchNorm1d(h_dim))
            self.layers.append(activation)
        self.layers.append(nn.Linear(h_dim, out_dim))
        
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class MetaLayer(torch.nn.Module):
    r"""A meta layer for building any kind of graph network, inspired by the
    `"Relational Inductive Biases, Deep Learning, and Graph Networks"
    <https://arxiv.org/abs/1806.01261>`_ paper.
    """
    def __init__(self, edge_model=None, node_model=None):
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model

        self.edge_eps = nn.Parameter(torch.Tensor([0]))
        self.node_eps = nn.Parameter(torch.Tensor([0]))

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """"""
        if self.edge_model is not None:
            edge_attr = (1 + self.edge_eps) * edge_attr + self.edge_model(x, edge_attr, edge_index)
        if self.node_model is not None:
            x = (1 + self.node_eps) * x + self.node_model(x, edge_index, edge_attr, batch)

        return x, edge_attr


class EdgeModel(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super(EdgeModel, self).__init__()
        self.edge = Lin(hidden_dim, hidden_dim)
        self.node_in = Lin(hidden_dim, hidden_dim, bias=False)
        self.node_out = Lin(hidden_dim, hidden_dim, bias=False)
        self.mlp = MLP(hidden_dim, hidden_dim, n_layers)

    def forward(self, x, edge_attr, edge_index):
        # source, target: [2, E], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs (we don't have any of these yet)
        # batch: [E] with max entry B - 1.

        f_ij = self.edge(edge_attr)
        f_i = self.node_in(x)
        f_j = self.node_out(x)
        row, col = edge_index

        out = F.relu(f_ij + f_i[row] + f_j[col])
        return self.mlp(out)


class NodeModel(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = MLP(hidden_dim, hidden_dim, n_layers)
        self.node_mlp_2 = MLP(hidden_dim, hidden_dim, n_layers)

    def forward(self, x, edge_index, edge_attr, batch):
        # x: [N, h], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u] (N/A)
        # batch: [N] with max entry B - 1.
        # source, target = edge_index
        _, col = edge_index
        out = self.node_mlp_1(edge_attr)
        out = scatter_sum(out, col, dim=0, dim_size=x.size(0))
        return self.node_mlp_2(out)


class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=300, depth=3, n_layers=2):
        super(GNN, self).__init__()
        self.depth = depth
        self.node_init = MLP(node_dim, hidden_dim, n_layers)
        self.edge_init = MLP(edge_dim, hidden_dim, n_layers)
        self.update = MetaLayer(EdgeModel(hidden_dim, n_layers), NodeModel(hidden_dim, n_layers))

    def forward(self, x, edge_index, edge_attr):

        x = self.node_init(x)
        edge_attr = self.edge_init(edge_attr)
        for _ in range(self.depth):
            x, edge_attr = self.update(x, edge_index, edge_attr)
        return x, edge_attr
