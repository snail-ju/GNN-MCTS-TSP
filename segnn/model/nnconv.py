import torch.nn as nn
from torch.nn import Sequential as Seq
from torch_geometric.nn.conv import NNConv
from torch_geometric.nn.conv import MessagePassing


class NNConvLayer(MessagePassing):
    def __init__(self, node_channels, edge_channels, out_channels, aggr='add'):
        super(NNConvLayer, self).__init__()
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.out_channels = out_channels

        self.aggr = aggr
        self.mlp_edge = Seq(nn.Linear(self.edge_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(),
                            nn.Linear(out_channels, node_channels * out_channels))
        self.mlp_output = nn.ReLU()
        self.nn_conv = NNConv(self.node_channels, self.out_channels, self.mlp_edge, aggr=self.aggr)

    def forward(self, node_attr, edge_index, edge_attr):
        return self.mlp_output(self.nn_conv(node_attr, edge_index, edge_attr))


class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, embed_dim, layer):
        super(GNN, self).__init__()
        self.layer = layer
        # Layers
        self.gnn_layers = nn.ModuleList([NNConvLayer(node_dim, edge_dim, embed_dim)])
        for _ in range(1, layer):
            self.gnn_layers.append(NNConvLayer(embed_dim, edge_dim, embed_dim))

    def forward(self, node_attr, edge_index, edge_attr):
        for i in range(self.layer):
            node_attr = self.gnn_layers[i](node_attr, edge_index, edge_attr)

        return node_attr
