import torch
import torch.nn as nn
from base import BaseModel
# from model.gnn import GNN

# from model.gnn_sigma import GNN
# from model.gnn_share import GNN
# from model.basic_gnn import GNN
# from model.s2v import GNN
# import torch_geometric.nn.conv.nn_conv as GNN
# from model.NNConv import GNN
# from model.CGConv import GNN
from model.gcn import GNN


class TSPModel(BaseModel):
    def __init__(self, node_dim, edge_dim, embed_dim, conv_dim, graph_size, layer):
        super(TSPModel, self).__init__()
        self.graph_size = graph_size
        self.embed_dim = embed_dim

        self.gnn_embed = GNN(node_dim, edge_dim, embed_dim, layer)

        self.lin1 = nn.Linear(embed_dim, conv_dim)
        self.lin2 = nn.Linear(conv_dim, conv_dim)

        self.bn1 = nn.BatchNorm1d(conv_dim)
        self.bn2 = nn.BatchNorm1d(conv_dim)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.LogSoftmax(dim=1)

    def forward(self, data):
        node_attr, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        embedded = self.gnn_embed(node_attr, edge_index, edge_attr)
        x = self.act1(self.lin1(embedded))
        x = self.act2(self.lin2(x))
        x = x.contiguous().view(-1, self.graph_size, self.embed_dim)
        x = torch.sum(x, dim=2)
        probs = self.act3(x)

        return probs
