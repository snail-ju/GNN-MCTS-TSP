import math
import time
import torch
import numpy as np
from args import parser
from utils import load_data
from env.tsp_env import TSPEnv
from model.policy import TSPNetwork


def eval(env, net, states, device=0):
    obs = env.get_obs_for_states(states)
    priors = net.step(obs, device)
    priors = priors.detach().cpu().numpy()

    return priors


def beam_search(env, net, state, k=5):
    sequences = [[state, 0.0]]
    while True:
        all_candidates = []
        # 遍历sequences中的每个元素
        for idx, sequence in enumerate(sequences):
            state, score = sequences[idx]
            priors = eval(env, net, [state])[0]
            priors = priors[list(state['ava_action'])]

            for p, action in zip(priors, state['ava_action']):
                all_candidates.append([env.next_state(state, action), score - math.log(p + 1e-8)])

        sequences = []
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        for i in range(min(k, len(ordered))):
            sequences.append(ordered[i])

        if env.is_done_state(sequences[0][0]):
            return env.get_return(sequences[0][0])


if __name__ == '__main__':
    args = parser.parse_args()
    test_graphs = load_data("data/{}/tsp{}.txt".format(args.graph_type, args.graph_size), args.graph_size)
    network = TSPNetwork(node_dim=args.node_dim, edge_dim=args.edge_dim, embed_dim=args.embed_dim,
                         hidden_dim=args.hidden_dim, graph_size=args.graph_size, layer=args.n_layer)

    device_id = 0
    network.to(device_id)
    network.eval()
    if args.load:
        saved_state = torch.load(
            "{}/{}/tsp{}.pth".format(args.load_model_dir, args.graph_type, args.graph_size),
            map_location=lambda storage, loc: storage)
        network.load_paras(saved_state)
        print("Load model successfully ~~")

    beam_width = [1, 5, 10, 15, 20]
    for b_w in beam_width:
        s_t = time.time()
        path_lens = []
        for idx, graph in enumerate(test_graphs):
            env = TSPEnv(graph, args)
            state = env.initial_state()
            path_len = beam_search(env, network, state, b_w)
            path_lens.append(path_len)

        time.time() - s_t
        print(np.mean(path_lens), (time.time() - s_t) / len(test_graphs) * 1000)
