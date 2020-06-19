import os
import time
import torch
import torch.multiprocessing as mp

from args import parser
from agent import Agent
from utils import load_data
from model.policy import TSPNetwork
from model.PolicyNet import PolicyNet
from hyperopt import hp, fmin, rand, tpe, space_eval
from analysis import statistic

os.environ["OMP_NUM_THREADS"] = "1"


def objective(virtual_loss):
    args = parser.parse_args()
    args.virtual_loss = virtual_loss
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

    test_graphs = load_data("data/{}/tsp{}.txt".format(args.graph_type, args.graph_size), args.graph_size)
    # network = TSPNetwork(node_dim=args.node_dim, edge_dim=args.edge_dim, embed_dim=args.embed_dim,
    #                      hidden_dim=args.hidden_dim, graph_size=args.graph_size, layer=args.n_layer)
    network = PolicyNet(node_dim=args.node_dim, edge_dim=args.edge_dim, embed_dim=args.embed_dim,
                        hidden_dim=args.hidden_dim, graph_size=args.graph_size, layer=args.n_layer)

    if args.load:
        saved_state = torch.load(
            "{}/{}/tsp{}.pth".format(args.load_model_dir, args.graph_type, args.graph_size),
            map_location=lambda storage, loc: storage)['state_dict']
        network.load_paras(saved_state)
        print("Load model successfully ~~")

    processes = []
    tasks_num = len(test_graphs) // args.n_worker
    extra_num = len(test_graphs) % args.n_worker
    print(args.n_worker)
    for idx in range(args.n_worker):
        if idx == args.n_worker - 1:
            graphs = test_graphs[idx * tasks_num: (idx + 1) * tasks_num + extra_num]
        else:
            graphs = test_graphs[idx * tasks_num: (idx + 1) * tasks_num]

        agent = Agent(idx, args, graphs, network)
        p = mp.Process(target=agent.run)
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()

    mean_path_len = statistic()
    with open('paras.txt', 'a') as f:
        f.write("{} {}\n".format(virtual_loss, mean_path_len))
    return mean_path_len


if __name__ == '__main__':
    # space = {
    #     # 'a': hp.uniform('a', 0.0, 1.0),  # 均匀分布（0~1）之间搜索
    #     'virtual_loss': hp.quniform('virtual_loss', 0, 100, 1),  # 均匀分布(0~100)之间搜索整数，最小间隔为1
    #     # 'c': hp.loguniform('c', np.log(0.01), np.log(1)),  # 对数空间(0.01~1)之间搜索
    #     # 'd': hp.choice('d', ["foo", "bar"]),  # 选项搜索，选择["foo", "bar"]之一的选项
    # }
    mp.set_start_method('spawn')
    # best = fmin(fn=objective,  # 目标函数
    #             space=hp.quniform('virtual_loss', 1, 20, 1),  # 搜索空间
    #             algo=tpe.suggest,  # 搜索算法(tpe),随机搜索是rand.suggest
    #             max_evals=100)  # 最大搜索次数
    for i in [3, 4, 5, 9, 11, 19]:
        print(objective(i))  # 最终返回的best就是最优化的参数
