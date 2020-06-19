import os
import os.path as osp
import torch
import numpy as np
from env.graph import BaseGraph
from env.simulator import TspSimulator
from tqdm import tqdm


class DataGenerator:
    def __init__(self, config):
        super(DataGenerator, self).__init__()
        self.config = config

        # Data file path
        self.points_dir = os.path.join('data', self.config['data_loader']['data']['graph_type'],
                                       'tsp{}'.format(self.config['arch']['args']['graph_size']),
                                       'tsp{}_train.txt'.format(self.config['arch']['args']['graph_size']))
        self.paths_dir = os.path.join('data', self.config['data_loader']['data']['graph_type'],
                                      'tsp{}'.format(self.config['arch']['args']['graph_size']),
                                      'tsp{}_train_path.txt'.format(self.config['arch']['args']['graph_size']))

    def generate_data(self, graph, path):

        simulator = TspSimulator(self.config, graph, path)
        data = simulator.play()

        return data

    def run(self, dir):
        graphs, paths = self.load_data()
        i = 0
        for (graph, path) in tqdm(zip(graphs, paths), total=len(graphs), desc='Generate Data'):
            data = self.generate_data(graph, path)
            for d in data:
                torch.save(d, osp.join(dir, 'data_{}.pt'.format(i)))
                i += 1

    def load_data(self):
        graphs = []
        paths = []
        vertex_number = self.config['arch']['args']['graph_size']
        path_content = None
        if os.path.exists(self.paths_dir):
            path_content = np.genfromtxt(self.paths_dir)

        if os.path.exists(self.points_dir):
            file_content = np.genfromtxt(self.points_dir)
            points_content = np.delete(file_content, [vertex_number * 2, vertex_number * 3 + 1], axis=1)
            if path_content is None:
                path_content = points_content[:, vertex_number * 2:]
            # path_content = np.genfromtxt(self.paths_dir)

            for idx, c in enumerate(tqdm(points_content, desc='Load Graph...')):
                vertex_coordinate = c[0:vertex_number * 2].reshape(vertex_number, 2)
                path = [int(i) for i in path_content[idx]]
                if vertex_number in path:
                    path = [int(i) - 1 for i in path]

                g = BaseGraph(vertex_number, vertex_coordinate)
                g.init_graph(self.config['data_loader']['data']['knn'])
                graphs.append(g)
                paths.append(path)

                if len(graphs) == self.config['data_loader']['data']['graph_num']:
                    break

        return graphs, paths
