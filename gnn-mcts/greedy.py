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


def greedy_search(env, net, state):
    state = [state]
    while not env.is_done_state(state[0]):
        action_prob = eval(env, net, state)[0]
        action_prob[state[0]['tour']] = 0
        action = np.argmax(action_prob)
        state[0] = env.next_state(state[0], action)

    return env.get_return(state[0])


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

    s_t = time.time()
    path_lens = []
    for idx, graph in enumerate(test_graphs):
        env = TSPEnv(graph, args)
        state = env.initial_state()
        path_len = greedy_search(env, network, state)
        path_lens.append(path_len)
        print(idx)

    time.time() - s_t
    print(np.mean(path_lens), (time.time() - s_t) / len(test_graphs) * 1000)
