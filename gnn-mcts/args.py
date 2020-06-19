from __future__ import print_function, division
import argparse

parser = argparse.ArgumentParser(description='TSP')

# Model
parser.add_argument('--node-dim', type=int, default=6, help='dimension of node attribute')
parser.add_argument('--edge-dim', type=int, default=4, help='dimension of edge attribute')
parser.add_argument('--embed-dim', type=int, default=128, help='dimension of embedded feature')
parser.add_argument('--hidden-dim', type=int, default=128, help='dimension of hidden layers in policy network')
parser.add_argument('--n-layer', type=int, default=3, help='number of layers in the graph neural network')

# Search
parser.add_argument('--n-playout', type=int, default=800, help='number of playout in the mcts')
parser.add_argument('--c-puct', type=float, default=1.3, help=' c_puct value in the selection phase of mcts')
parser.add_argument('--n-parallel', type=int, default=32, help='number of parallel threed in the mcts')
parser.add_argument('--virtual-loss', type=float, default=2, help='value of virtual loss in the mcts')
parser.add_argument('--q-init', type=float, default=5, help='initial Q value in the mcts')

# GPU
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--gpu-ids', type=int, default=-1, nargs='+', help='GPUs to use [-1 CPU only] (default: -1)')

# Data
parser.add_argument('--graph-type', type=str, default='random', help='type of the graph (random or clustered)')
parser.add_argument('--graph-size', type=int, default=20, help="size of the instance")
parser.add_argument('--n-neighbor', type=int, default=10, help="number of node neighbor")

# Train
parser.add_argument('--load', default=True, help='load a trained model')
parser.add_argument('--load-model-dir', default='pre_trained_model/segnn',
                    help='folder to load trained models from')
parser.add_argument('--n-worker', type=int, default=25, help='how many training processes to use (default:16)')
parser.add_argument('--result-dir', type=str, default='result', help='dir to save tours')
