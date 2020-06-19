import torch
import numpy as np
from env.graph import BaseGraph
from torch_geometric.data import Data


class TspSimulator:
    def __init__(self, config, graph: BaseGraph, opt_path):
        self.config = config
        self.graph = graph
        self.opt_path = opt_path

    def init(self):
        self.obj_path = []
        self.visited = np.zeros(self.graph.ver_num)

        self.data_list = []

    def move(self, vertex):
        self.visited[vertex] = 1
        self.obj_path.append(vertex)

    def play(self):
        self.init()

        for idx, vertex in enumerate(self.opt_path):
            if idx == 0:
                self.move(vertex)
            else:
                self.save_state(vertex)
                self.move(vertex)

        return self.data_list

    def save_state(self, move):
        node_tag = np.zeros((self.graph.ver_num, self.config['arch']['args']['node_dim']), dtype=np.float)
        node_tag[:, 0] = self.visited
        node_tag[:, 1:3] = self.graph.ver_coor
        node_tag[:, 3:4] = 1  # [s_node:tag]
        node_tag[:, 4:6] = \
            self.graph.ver_coor[self.obj_path[0]]  # [s_node:x,y]

        edge_tag = np.zeros(
            (self.graph.ver_num, self.config['data_loader']['data']['knn'], self.config['arch']['args']['edge_dim']),
            dtype=np.float)
        edge_tag[:, :, 0] = self.graph.knn_mat
        edge_tag[:, :, 1:2] = 1
        edge_tag[:, :, 2:4] = self.graph.ver_coor[self.obj_path[-1]]

        node_tag = torch.tensor(node_tag, dtype=torch.float)
        edge_tag = torch.tensor(edge_tag, dtype=torch.float).view(-1, self.config['arch']['args']['edge_dim'])
        edge_index = torch.tensor(self.graph.edge_index, dtype=torch.long)
        y = torch.tensor([move], dtype=torch.long)
        data = Data(x=node_tag, edge_index=edge_index, edge_attr=edge_tag, y=y)
        self.data_list.append(data)
