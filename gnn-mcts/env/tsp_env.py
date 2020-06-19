import gym
import copy
import torch
import numpy as np
from env.graph import Graph
from torch_geometric.data import Data
from env.static_env import StaticEnv


class TSPEnv(gym.Env, StaticEnv):

    def __init__(self, graph: Graph, args):
        self.graph = graph
        self.args = args

        self.step_idx = 0
        self.n_actions = graph.ver_num
        self.tour = []
        self.visited = [0 for _ in range(self.n_actions)]
        self.ava_actions = set((range(self.n_actions)))

    def reset(self):
        self.step_idx = 0
        self.tour = []
        self.visited = [0 for _ in range(self.graph.ver_num)]
        self.ava_actions = set((range(self.graph.ver_num)))

        state, r, done = self.step(0)

        return state, r, done

    def step(self, action):
        self.step_idx += 1
        self.ava_actions.remove(action)
        self.tour.append(action)
        self.visited[action] = 1
        state = {
            'step_idx': self.step_idx,
            'n_actions': self.n_actions,
            'tour': self.tour,
            'visited': self.visited,
            'ava_action': self.ava_actions,
            'graph': self.graph,
            'args': self.args,
        }

        done = (self.step_idx == self.n_actions)

        return state, 0, done

    def render(self, mode='human'):
        pass

    @staticmethod
    def next_state(state, action):
        if state['step_idx'] == state['n_actions']:
            return state

        tour = copy.copy(state['tour'])
        visited = copy.copy(state['visited'])
        ava_action = copy.copy(state['ava_action'])

        tour.append(action)
        visited[action] = 1
        ava_action.remove(action)
        step_idx = state['step_idx'] + 1
        n_actions = state['n_actions']

        next_state = {
            'step_idx': step_idx,
            'n_actions': n_actions,
            'tour': tour,
            'visited': visited,
            'ava_action': ava_action,
            'graph': state['graph'],
            'args': state['args']
        }

        return next_state

    @staticmethod
    def is_done_state(state, step_idx=None):
        return state['n_actions'] == state['step_idx']

    def initial_state(self):
        state, r, done = self.reset()

        return state

    @staticmethod
    def get_obs_for_states(states):
        return_data = []
        for state in states:
            ver_num = state['graph'].ver_num
            node_attr = np.zeros((ver_num, state['args'].node_dim), dtype=np.float)
            node_attr[:, 0] = state['visited']
            node_attr[:, 1:3] = state['graph'].ver_coo
            node_attr[:, 3] = 1
            node_attr[:, 4:6] = state['graph'].ver_coo[state['tour'][0]]

            edge_attr = np.zeros((ver_num, state['args'].n_neighbor, state['args'].edge_dim), dtype=np.float)
            edge_attr[:, :, 0] = state['graph'].knn_mat
            edge_attr[:, :, 1] = 1
            edge_attr[:, :, 2:4] = state['graph'].ver_coo[state['tour'][-1]]

            node_attr = torch.tensor(node_attr, dtype=torch.float)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, state['args'].edge_dim)
            edge_index = torch.tensor(state['graph'].edge_index, dtype=torch.long)
            data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)

            return_data.append(data)

        return return_data

    # @staticmethod
    # def get_obs_for_states(states):
    #     node_attrs = []
    #     edge_attrs = []
    #     n2n_sps = []
    #
    #     for state in states:
    #         ver_num = state['graph'].ver_num
    #         node_attr = np.zeros((ver_num, state['args'].node_dim), dtype=np.float)
    #         node_attr[:, 0] = state['visited']
    #         node_attr[:, 1:3] = state['graph'].ver_coo
    #         node_attr[:, 3] = 1
    #         node_attr[:, 4:6] = state['graph'].ver_coo[state['tour'][0]]
    #
    #         edge_attr = np.zeros((ver_num, ver_num, state['args'].edge_dim), dtype=np.float)
    #         edge_attr[:, :, 0] = state['graph'].dis_mat
    #         edge_attr[:, :, 1] = 1
    #         edge_attr[:, :, 2:4] = state['graph'].ver_coo[state['tour'][-1]]
    #         n2n_sp = np.ones(shape=(ver_num, ver_num), dtype=np.float)
    #
    #         node_attr = np.expand_dims(node_attr, axis=0)
    #         edge_attr = np.expand_dims(edge_attr, axis=0)
    #         n2n_sp = np.expand_dims(n2n_sp, axis=0)
    #
    #         node_attrs.append(node_attr)
    #         edge_attrs.append(edge_attr)
    #         n2n_sps.append(n2n_sp)
    #
    #     return np.vstack(node_attrs), np.vstack(edge_attrs), np.vstack(n2n_sps)

    def get_return(self, state):
        tour_len = state['graph'].compute_path_len(state['tour'])
        return tour_len
