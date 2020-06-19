import torch
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


class Graph:
    def __init__(self, ver_num, ver_coo, k_num=10):
        self.ver_num = ver_num
        self.ver_coo = ver_coo
        self.k_num = k_num

    def init(self):
        self.edge_index, self.knn_mat = self.cal_k_neighbor(self.k_num)

    def cal_dis_mat(self, vertex_i, vertex_j):
        return np.linalg.norm(self.ver_coo[vertex_i] - self.ver_coo[vertex_j])

    def cal_k_neighbor(self, k_n):
        source_ver = []
        target_ver = []
        knn_mat = np.zeros((self.ver_num, k_n))
        self.dis_mat = squareform(pdist(self.ver_coo, metric='euclidean'))

        value_idx = torch.topk(torch.tensor(self.dis_mat, dtype=torch.float), k_n + 1, largest=False)
        values, indices = value_idx[0].numpy(), value_idx[1].numpy()

        for i in range(self.ver_num):
            source_ver.extend([i for _ in range(k_n)])
            target_ver.extend(indices[i][1:])
            knn_mat[i] = values[i][1:]

        return [source_ver, target_ver], knn_mat

    def compute_path_len(self, path):
        path_len = 0.0
        for index in range(len(path)):
            path_len += self.cal_dis_mat(
                path[index % len(path)],
                path[(index + 1) % len(path)])

        return round(path_len, 3)
