import torch.nn as nn
from torch.nn import Sequential as Seq
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter_


class EdgeConv(MessagePassing):

    def __init__(self, node_channels, edge_channels, out_channels, aggr='add'):
        super(EdgeConv, self).__init__()

        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.lin_node = nn.Linear(node_channels, out_channels)
        self.lin_edge = nn.Linear(edge_channels, out_channels)
        self.lin_neighbor = nn.Linear(node_channels, out_channels)

        self.bn_node = nn.BatchNorm1d(out_channels)
        self.bn_edge = nn.BatchNorm1d(out_channels)
        self.bn_neighbor = nn.BatchNorm1d(out_channels)

        self.mlp = Seq(nn.ReLU(), nn.Linear(out_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(),
                       nn.Linear(out_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU())

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
        self.mlp = Seq(nn.Linear(node_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU())
        # Layers
        # self.gnn_layers = nn.ModuleList([EdgeConv(node_dim, edge_dim, embed_dim)])
        self.gnn_layers = nn.ModuleList()
        for _ in range(layer):
            self.gnn_layers.append(EdgeConv(embed_dim, edge_dim, embed_dim))

    def forward(self, node_attr, edge_index, edge_attr):
        for i in range(self.layer):
            node_attr = self.mlp(node_attr) if i == 0 else node_attr
            node_attr = self.gnn_layers[i](node_attr, edge_index, edge_attr)

        return node_attr
