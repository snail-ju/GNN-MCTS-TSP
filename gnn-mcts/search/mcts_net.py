# -*- coding: utf-8 -*-
import math
import random
import numpy as np


class TreeNode:
    def __init__(self, state, parent, prior_p, q_init=5):
        self.state = state
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = q_init
        self._u = 0
        self._P = prior_p
        self.q_init = q_init
        self.max_Q = q_init
        self.min_Q = 100.0
        self.n_vlosses = 0

    def expand(self, actions, priors, states):
        for (action, prob, state) in zip(actions, priors, states):
            if action not in self._children:
                self._children[action] = TreeNode(state, self, prob, self.q_init)

    def select(self, c_puct):
        mean_Q = np.mean([node._Q for node in self._children.values()])
        return max(self._children.items(),
                   key=lambda item: item[1].get_value(c_puct, self.max_Q, self.min_Q, mean_Q))

    def add_virtual_loss(self, virtual_loss):
        if self._parent:
            self._parent.add_virtual_loss(virtual_loss)
        self.n_vlosses += 1
        self._n_visits += virtual_loss

    def revert_virtual_loss(self, virtual_loss):
        if self._parent:
            self._parent.add_virtual_loss(virtual_loss)
        self.n_vlosses -= 1
        self._n_visits -= virtual_loss

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q = leaf_value if leaf_value < self._Q else self._Q

        self.max_Q = leaf_value if leaf_value > self.max_Q else self.max_Q
        self.min_Q = leaf_value if leaf_value < self.min_Q else self.min_Q

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct, max_value, min_value, mean_value):
        self._u = (c_puct * self._P * math.sqrt(self._parent._n_visits + 1) / (1 + self._n_visits))
        if max_value - min_value == 0:
            return -self._Q + self._u
        else:
            return -(self._Q - mean_value) / (max_value - min_value) + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS:
    def __init__(self, env, device_id, net, c_puct=5, n_playout=400, n_parallel=16, virtual_loss=20, q_init=5):
        self.env = env
        self.device_id = device_id
        self._net = net
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.n_parallel = n_parallel
        self.virtual_loss = virtual_loss
        self.q_init = q_init
        self._root = None

    def initialize_search(self):
        state = self.env.initial_state()
        self._root = TreeNode(state, None, 1.0, self.q_init)

    def select_leaf(self):
        current = self._root
        while True:
            if current.is_leaf():
                break
            _, current = current.select(self._c_puct)

        return current

    def _playout(self, num_parallel):
        leaves = []
        failsafe = 0
        while len(leaves) < num_parallel and failsafe < num_parallel * 2:
            failsafe += 1
            leaf = self.select_leaf()
            if self.env.is_done_state(leaf.state):
                leaf_value = self.env.get_return(leaf.state)
                leaf.update_recursive(leaf_value)

            else:
                leaf.add_virtual_loss(self.virtual_loss)
                leaves.append(leaf)

        if leaves:
            # revert_virtual_loss
            for leaf in leaves:
                leaf.revert_virtual_loss(self.virtual_loss)
            # Cal priors
            priors = self._eval([leaf.state for leaf in leaves])
            # priors = np.ones((len(leaves), leaves[0].state['graph'].ver_num))
            # Cla values
            values = self.evaluate_leaf(leaves)

            for idx, (leaf, ps, value) in enumerate(zip(leaves, priors, values)):
                # update_value
                leaf.update_recursive(value)
                # expand node
                prior = ps[list(leaf.state['ava_action'])]
                states = [self.env.next_state(leaf.state, act) for act in leaf.state['ava_action']]
                leaf.expand(leaf.state['ava_action'], prior, states)

    def evaluate_leaf(self, leaves):
        # result = []
        # for leaf in leaves:
        #     result.append(self.beam_search(leaf.state))
        #     print(result[-1])
        #     time.txt.sleep(10)
        # return result

        return self.value_func(leaves)

        # result = []
        # for leaf in leaves:
        #     result.append(self.random_rollout(leaf.state))
        # return result

    def random_rollout(self, state):
        while not self.env.is_done_state(state):
            state = self.env.next_state(state, random.choice(list(state['ava_action'])))

        return self.env.get_return(state)

    def beam_search(self, state, k=5):
        sequences = [[state, 0.0]]
        while True:
            all_candidates = []
            # 遍历sequences中的每个元素
            for idx, sequence in enumerate(sequences):
                state, score = sequences[idx]
                priors = self._eval([state])[0]
                priors = priors[list(state['ava_action'])]

                for p, action in zip(priors, state['ava_action']):
                    all_candidates.append([self.env.next_state(state, action), score - math.log(p + 1e-8)])

            sequences = []
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            for i in range(min(k, len(ordered))):
                sequences.append(ordered[i])

            if self.env.is_done_state(sequences[0][0]):
                return self.env.get_return(sequences[0][0])

    def value_func(self, leaves):
        states = []
        max_eval_count = 0
        for leaf in leaves:
            states.append(leaf.state)
            max_eval_count = max(max_eval_count, leaf.state['n_actions'] - leaf.state['step_idx'])

        while max_eval_count > 0:
            action_probs = self._eval([state for state in states])

            for idx, action_prob in enumerate(action_probs):
                action_prob[states[idx]['tour']] = 0
                action = np.argmax(action_prob)
                states[idx] = self.env.next_state(states[idx], action)

            max_eval_count -= 1

        return [self.env.get_return(state) for state in states]

    def _eval(self, states):
        obs = self.env.get_obs_for_states(states)
        priors = self._net.step(obs, self.device_id)
        priors = priors.detach().cpu().numpy()

        return priors

    def get_move_values(self):
        current_simulations = self._root._n_visits
        while self._root._n_visits < self._n_playout + current_simulations:
            self._playout(self.n_parallel)

        act_values_states = [(act, node._Q, node.state) for act, node in self._root._children.items()]
        return zip(*act_values_states)

    def update_with_move(self, last_move, last_state):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(last_state, None, 1.0, self.q_init)


class MCTSPlayer:
    def __init__(self, env, device_id, net, c_puct=5, n_playout=400, n_parallel=16, virtual_loss=20, q_init=5):
        self.mcts = MCTS(env, device_id, net, c_puct, n_playout, n_parallel, virtual_loss, q_init)
        self.mcts.initialize_search()

    def get_action(self):
        acts, values, states = self.mcts.get_move_values()
        idx = np.argmin(values)
        self.mcts.update_with_move(acts[idx], states[idx])

        return acts[idx], values[idx]
