import torch
import torch.nn as nn
from model.segnn import GNN
# from model.segnn_share import GNN
# from model.segnn_sigma import GNN
# from model.segnn_x import GNN
# from model.gen import GNN
# from model.gnn import GNN
# from model.cgconv import GNN
# from model.nnconv import GNN
# from model.gcn import GNN
from torch_geometric.data import DataLoader


class PolicyModel(nn.Module):
    def __init__(self, node_dim, edge_dim, embed_dim, hidden_dim, class_num, layer):
        super(PolicyModel, self).__init__()

        self.class_num = class_num
        self.embed_dim = embed_dim

        self.gnn_embed = GNN(node_dim, edge_dim, embed_dim, layer)

        self.lin1 = nn.Linear(embed_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.Softmax(dim=-1)

    def forward(self, data):
        node_attr, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        embedded = self.gnn_embed(node_attr, edge_index, edge_attr)
        x = self.act1(self.lin1(embedded))
        x = self.act2(self.lin2(x))
        x = x.contiguous().view(-1, self.class_num, self.embed_dim)
        logits = torch.sum(x, dim=2)
        probs = self.act3(logits)

        return probs, logits


class TSPNetwork:
    def __init__(self, node_dim, edge_dim, embed_dim, hidden_dim, graph_size, layer):
        self.policy = PolicyModel(node_dim, edge_dim, embed_dim, hidden_dim, graph_size, layer)

    def step(self, data, device):
        """
        Returns policy for given observations.
        :return: Policy estimate [N, n_actions] for the given observations.
        """
        for data in DataLoader(data, batch_size=len(data)):
            data = data.to(device)
        pi, logits = self.policy(data)

        return pi

    def load_paras(self, state_dict):
        self.policy.load_state_dict(state_dict)

    def get_paras(self):
        return self.policy.state_dict()

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def to(self, device_id):
        if device_id >= 0:
            self.policy = self.policy.to(device_id)
        else:
            self.policy = self.policy.cpu()
