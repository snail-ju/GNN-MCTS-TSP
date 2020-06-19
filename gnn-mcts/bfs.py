import os
import copy
import time

import torch
import torch.multiprocessing as mp
import numpy as np

from args import parser
from utils import load_data
from model.policy import TSPNetwork
from env.tsp_env import TSPEnv

os.environ["OMP_NUM_THREADS"] = "1"

time_limit = 10
start_time = time.time()


def eval(states, env, net, device_id):
    obs = env.get_obs_for_states(states)
    priors = net.step(obs, device_id)
    return priors[0]


def search(net, env, device_id, bsf_q, best_tour, q_lock, t_lock):
    gpu_start_time = time.time()
    net = copy.deepcopy(net)
    net.to(device_id)
    gpu_time = time.time() - gpu_start_time

    while time.time() - start_time - gpu_time < time_limit:
        with q_lock:
            if len(bsf_q) == 0:
                continue

            q_item = bsf_q.pop(np.random.randint(0, len(bsf_q)))

        if env.is_done_state(q_item):
            with t_lock:
                best_tour.value = min(best_tour.value, env.get_return(q_item))
                continue

        prior = eval([q_item], env, net, device_id)
        prior = prior[list(q_item['ava_action'])]
        prior_index = torch.topk(prior, min(3, len(q_item['ava_action'])))

        index = prior_index[1].detach().cpu().numpy()
        actions = [list(q_item['ava_action'])[idx] for idx in index]
        states = [env.next_state(q_item, act) for act in actions]
        with q_lock:
            bsf_q.extend(states)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device_id = 0

    net = TSPNetwork(node_dim=args.node_dim, edge_dim=args.edge_dim, embed_dim=args.embed_dim,
                     hidden_dim=args.hidden_dim, graph_size=args.graph_size, layer=args.n_layer)
    saved_state = torch.load(
        "{}/{}/tsp{}.pth".format(args.load_model_dir, args.graph_type, args.graph_size),
        map_location=lambda storage, loc: storage)
    net.load_paras(saved_state)

    test_graphs = load_data("data/{}/tsp{}.txt".format(args.graph_type, args.graph_size), args.graph_size)
    tour_length = []
    for graph in test_graphs:
        env = TSPEnv(graph, args)
        with mp.Manager() as manager:
            bsf_q = manager.list()
            best_tour = manager.Value('f', 100.0)

            q_lock = manager.Lock()
            t_lock = manager.Lock()

            bsf_q.append(env.initial_state())
            processes = [mp.Process(target=search, args=(net, env, device_id, bsf_q, best_tour, q_lock, t_lock)) for _
                         in range(8)]

            for p in processes:
                p.start()
            for p in processes:
                p.join()

            if best_tour.value <= 50.0:
                tour_length.append(best_tour.value)
                
            print(best_tour.value)
    print(np.mean(tour_length))
    print(len(tour_length))
