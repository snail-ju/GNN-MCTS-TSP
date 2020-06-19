import os
from model.policy import TSPNetwork
from env.tsp_env import TSPEnv
from env.simulator import Simulator
from search.mcts_net import MCTSPlayer
from utils import save_path
import time


class Agent:
    def __init__(self, rank, args, graphs, global_net):
        super(Agent, self).__init__()
        self.rank = rank
        self.args = args
        self.graphs = graphs
        self.global_net = global_net
        self.device_id = args.gpu_ids[self.rank % len(args.gpu_ids)]

        # Local Net
        self.local_net = TSPNetwork(node_dim=self.args.node_dim, edge_dim=self.args.edge_dim,
                                    embed_dim=self.args.embed_dim, hidden_dim=self.args.hidden_dim,
                                    graph_size=self.args.graph_size, layer=self.args.n_layer)

        self.local_net.load_paras(global_net.get_paras())
        self.local_net.to(self.device_id)
        self.local_net.eval()

    def run(self):
        for graph_idx, graph in enumerate(self.graphs):
            start_time = time.time()
            env = TSPEnv(graph, self.args)
            mcts_player = MCTSPlayer(env, self.device_id, self.local_net, c_puct=self.args.c_puct,
                                     n_playout=self.args.n_playout, n_parallel=self.args.n_parallel,
                                     virtual_loss=self.args.virtual_loss, q_init=self.args.q_init)
            simulator = Simulator(env)
            tour, tour_len = simulator.start(mcts_player, self.rank, graph_idx)

            dirs = os.path.join(self.args.result_dir, self.args.graph_type, "tsp{}".format(self.args.graph_size))
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            save_path(os.path.join(dirs, 'process_%s_path.txt' % str(self.rank)), tour, tour_len, graph_idx)
            print(time.time() - start_time)
