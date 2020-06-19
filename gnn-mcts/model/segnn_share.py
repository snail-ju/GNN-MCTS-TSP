import torch.nn as nn
from torch.nn import Sequential as Seq
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter_


class EdgeConv(MessagePassing):

    def __init__(self, lin_node, lin_edge, lin_neighbor, bn_node, bn_edge, bn_neighbor, mlp, aggr='add'):
        super(EdgeConv, self).__init__()

        self.node_channels = None
        self.edge_channels = None
        self.out_channels = None
        self.aggr = aggr

        self.lin_node = lin_node
        self.lin_edge = lin_edge
        self.lin_neighbor = lin_neighbor

        self.bn_node = bn_node
        self.bn_edge = bn_edge
        self.bn_neighbor = bn_neighbor

        self.mlp = mlp

    def forward(self, node_attr, edge_index, edge_attr):
        row, col = edge_index
        edge_attr = self.lin_edge(edge_attr)
        edge_attr = scatter_(self.aggr, edge_attr, row, dim_size=node_attr.size(0))
        edge_attr = self.bn_edge(edge_attr)

        h = self.lin_neighbor(node_attr)

        node_attr = self.lin_node(node_attr)
        node_attr = self.bn_node(node_attr)

        return self.propagate(edge_index, x=node_attr, h=h, edge_attr=edge_attr)

    def message(self, h_j):
        return h_j

    def update(self, aggr_out, x, edge_attr):
        return self.mlp(x + self.bn_neighbor(aggr_out) + edge_attr)

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.node_channels,
                                       self.edge_channels, self.out_channels)


class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, embed_dim, layer):
        super(GNN, self).__init__()
        self.layer = layer
        # Layers
        # share parameter
        lin_node_first = nn.Linear(node_dim, embed_dim)
        lin_node_other = nn.Linear(embed_dim, embed_dim)
        lin_edge = nn.Linear(edge_dim, embed_dim)
        lin_neighbor_first = nn.Linear(node_dim, embed_dim)
        lin_neighbor_other = nn.Linear(embed_dim, embed_dim)

        bn_node = nn.BatchNorm1d(embed_dim)
        bn_edge = nn.BatchNorm1d(embed_dim)
        bn_neighbor = nn.BatchNorm1d(embed_dim)

        mlp = Seq(nn.ReLU(), nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(),
                  nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU())

        self.gnn_layers = nn.ModuleList(
            [EdgeConv(lin_node_first, lin_edge, lin_neighbor_first, bn_node, bn_edge, bn_neighbor, mlp)])
        for _ in range(1, layer):
            self.gnn_layers.append(EdgeConv(lin_node_other, lin_edge, lin_neighbor_other, bn_node, bn_edge, bn_neighbor, mlp))

    def forward(self, node_attr, edge_index, edge_attr):
        for i in range(self.layer):
            node_attr = self.gnn_layers[i](node_attr, edge_index, edge_attr)

        return node_attr
