import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


class BaseGraph:
    def __init__(self, ver_num, ver_coor):
        self.ver_num = ver_num
        self.ver_coor = np.array(ver_coor)

    def cal_dis_mat(self):
        distA = pdist(self.ver_coor, metric='euclidean')
        return squareform(distA)

    def init_graph(self, k_num):
        self.dis_mat = self.cal_dis_mat()
        self.edge_index, self.knn_mat = self.cal_k_neighbor(k_num)

    def cal_k_neighbor(self, k_n):
        source_ver = []
        target_ver = []
        knn_mat = np.zeros((self.ver_num, k_n))

        v_idx = torch.topk(torch.tensor(self.dis_mat, dtype=torch.float), k_n + 1, largest=False)
        values = v_idx[0].detach().numpy()
        indices = v_idx[1].detach().numpy()

        for i in range(self.ver_num):
            source_ver.extend([i for _ in range(k_n)])
            target_ver.extend(indices[i][1:])
            knn_mat[i] = values[i][1:]

        return [source_ver, target_ver], knn_mat
