import os
import json
import numpy as np
from env.graph import Graph
from collections import OrderedDict
from datetime import datetime


def load_data(dir, graph_size):
    graphs = []
    if os.path.exists(dir):
        file_content = np.genfromtxt(dir)
        coordinates = np.delete(file_content, [graph_size * 2, graph_size * 3 + 1], axis=1)

    for idx, c in enumerate(coordinates):
        vertex_coordinate = c[0:graph_size * 2].reshape(graph_size, 2)
        g = Graph(graph_size, vertex_coordinate)
        g.init()
        graphs.append(g)

    return graphs


def save_path(file_path, tour, path_len, episode):
    file = open(file_path, 'a')
    file.write(str(episode))
    file.write(' ')

    for vertex in tour:
        file.write(str(vertex))
        file.write(' ')

    file.write(str(path_len))
    file.write('\n')
    file.close()


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
