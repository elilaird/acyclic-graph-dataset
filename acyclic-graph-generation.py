from utils import *

import os
from random import shuffle
from torch_geometric.data import Dataset
import argparse

# define arguments
parser = argparse.ArgumentParser(description='Is Acyclic Data Generation.\n Usage: python acyclic-graph-generation.py --max_cyclic_nodes 71 --max_acyclic_nodes 11 --num_graphs 10 \
                                 --seed 1 --save_dir ./data')
parser.add_argument('--max_cyclic_nodes', type=int, default=71, help='maximum number of nodes in cyclic graphs')
parser.add_argument('--max_acyclic_nodes', type=int, default=11, help='maximum number of nodes in acyclic graphs')
parser.add_argument('--num_graphs', type=int, default=100, help='number of graphs to generate')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--save_dir', type=str, default='data', help='directory to save generated graphs')
parser.add_argument('--gflownet-cyclic', type=str, default='models/gflownet_is_cyclic.pt', help='path to cyclic GFlowNet model (Default: models/gflownet_is_cyclic.pt)')
parser.add_argument('--gflownet-acyclic', type=str, default='models/gflownet_is_acyclic.pt', help='path to acyclic GFlowNet model (Default: models/gflownet_is_acyclic.pt)')


args = parser.parse_args()

# generate base dataset
dataset = create_is_acyclic(args.max_cyclic_nodes, args.max_acyclic_nodes)

# generate using gflownet is model is passed
if args.gflownet_cyclic is not None and args.gflownet_acyclic is not None:
    # check if cyclic/acyclic model file exists
    if not os.path.exists(args.gflownet_cyclic):
        print("Cyclic GFlowNet model file does not exist")
        exit(1)
    if not os.path.exists(args.gflownet_acyclic):
        print("Acyclic GFlowNet model file does not exist")
        exit(1)
    
    # load cyclic/acyclic graph gflownet
    gflownet_acyclic_generator = GFlowNet_Is_Acyclic(num_hidden=512, num_features=1)
    gflownet_acyclic_generator.load_state_dict(torch.load(args.gflownet_acyclic))
    gflownet_acyclic_generator.eval()

    gflownet_cyclic_generator = GFlowNet_Is_Acyclic(num_hidden=512, num_features=1)
    gflownet_cyclic_generator.load_state_dict(torch.load(args.gflownet_cyclic))
    gflownet_cyclic_generator.eval()


    dataset += generate_graphs(gflownet_acyclic_generator, args.num_graphs) + \
                generate_graphs(gflownet_cyclic_generator, args.num_graphs)
                
    
print("Total number of graphs: {}".format(len(dataset)))
shuffle(dataset)

# check if save directory exists, if not then create it
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# save dataset to file
torch.save(dataset, os.path.join(args.save_dir, "is_acyclic_{}.pt".format(args.num_graphs)))
