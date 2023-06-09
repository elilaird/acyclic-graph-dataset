import torch
import torch.nn as nn
from torch_geometric.utils import degree, from_networkx, to_networkx
from torch.distributions.categorical import Categorical
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import networkx as nx



## Networkx Graph Generation

def generate_cyclic_graphs(max_nodes):
    cycle_generators = {"cycle": nx.cycle_graph, "chordal_cycle": nx.chordal_cycle_graph, "wheel": nx.wheel_graph, "circular_ladder": nx.circular_ladder_graph}
    graphs = []
    for _, func in cycle_generators.items():
        for j in range(3, max_nodes+1):
            G = func(j)
            G_torch = from_networkx(G)
            G_torch.x = degree(G_torch.edge_index[0], G_torch.num_nodes).view(-1, 1).float()
            G_torch.y = torch.tensor([0])
            if 'name' in G_torch: G_torch.pop('name', None)
            if G_torch.num_nodes > 1:
                graphs.append(G_torch.clone())
    return graphs

def generate_acyclic_graphs(max_nodes):
    acyclic_generators = {"binomial_tree": nx.binomial_tree, "full_rary_tree": nx.full_rary_tree, "path": nx.path_graph}
    graphs = []
    for name, func in acyclic_generators.items():
        if name == "full_rary_tree":
            for j in range(1, max_nodes+1):
                for i in range(1, max_nodes+1):
                    G = func(j, i)
                    G_torch = from_networkx(G)
                    G_torch.x = degree(G_torch.edge_index[0], G_torch.num_nodes).view(-1, 1).float()
                    G_torch.y = torch.tensor([1])
                    if G_torch.num_nodes > 1:
                        graphs.append(G_torch.clone())
        else:
            for j in range(1, max_nodes+1):
                G = func(j)
                G_torch = from_networkx(G)
                G_torch.x = degree(G_torch.edge_index[0], G_torch.num_nodes).view(-1, 1).float()
                G_torch.y = torch.tensor([1])
                if G_torch.num_nodes > 1:
                    graphs.append(G_torch.clone())
    return graphs


def create_is_acyclic(max_cyclic=71, max_acyclic=11):
    cyclic = generate_cyclic_graphs(max_cyclic)
    acyclic = generate_acyclic_graphs(max_acyclic)
    return cyclic + acyclic


##  GFlowNet Utils
class GFlowNet_Is_Acyclic(torch.nn.Module):
    def __init__(self, num_hidden, num_features):
        super(GFlowNet_Is_Acyclic, self).__init__()
        torch.manual_seed(12345)

        self.num_features = num_features
        
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 128)
        self.relu = nn.LeakyReLU()

        # Predicts a new node/edge
        self.mlp = nn.Sequential(
            nn.Linear(128, num_hidden),
            nn.LeakyReLU(),
            nn.Linear(num_hidden, 4), # output is shape: num_nodes x 4 where columns are P_f_s, P_f_e, P_b_s, P_b_e
        )

        self.candidate_set = self._create_candidates(num_features)

        self.logZ = nn.Parameter(torch.ones(1))

    def _create_candidates(self, num_features):
        # Stop action is the first node in candidate list
        x = Data(
            x=torch.zeros(num_features + 1, num_features), # +1 adds stop action
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )

        for i in range(1, num_features + 1):
            x.x[i, i - 1] = 1

        return x

    def forward(self, x, edge_index):

        # Concatenate node embeddings and edge_index with candidate sets
        combined_x = torch.cat([x, self.candidate_set.x], dim=0)
        edge_index = torch.cat([edge_index, self.candidate_set.edge_index], dim=1)

        # Obtain node embeddings 
        out = self.conv1(combined_x, edge_index)
        out = self.relu(out)
        out = self.conv2(out, edge_index)
        out = self.relu(out)
        out = self.conv3(out, edge_index)

        # Obtain logits for new edge
        logits = self.mlp(out) # Num_Nodes x 4 

        # Mask for removing candidate set from starting nodes
        candidate_mask = torch.zeros_like(logits[:, 0]) # Num_Nodes x 1
        candidate_mask[:x.size(0)] = 1

        return logits[:, 0] * candidate_mask - 100 * (1 - candidate_mask), logits[:, 1], logits[:, 2] * candidate_mask - 100 * (1 - candidate_mask), logits[:, 3]

def check_edge(edge_index, new_edge):
    # Check if an edge is in the graph
    return bool(torch.all(torch.eq(edge_index, new_edge), dim=0).any())

def append_edge(edge_index, new_edge):
    return torch.cat((edge_index, new_edge), dim=1)

def create_initial_graph_is_acyclic(num_node_features):
    # Create initial graph with only the starting node
    g = Data(x=torch.zeros((1, num_node_features)), 
             edge_index=torch.zeros((2, 0), dtype=torch.long), 
             y=torch.torch.zeros((1, 1), dtype=torch.long),
        )
    
    
    g.x[0, 0] = 0
    g.y[0] = 0

    return g

def get_node_degree(G, node):
    # Returns the degree of a node in a graph
    return (G.edge_index[0] == node).sum().item()

def get_new_state_is_acyclic(G, action):
    # Takes an action in the form (starting_node, ending_node)
    start, end = action

    G_new = G.clone()

    # If end node is stop action, return graph
    if end == G.x.size(0):
        return G, True, True

    # If end node is new candidate, add it to the graph
    if end > G.x.size(0):
        # Create new node
        candidate_idx = end - G.x.size(0) - 1
        new_feature = torch.zeros(1, G_new.x.size(1))
        new_feature[0, 0] = 0 # set degree of new node to 1
        G_new.x = torch.cat([G_new.x, new_feature], dim=0)
        end = G_new.x.size(0) - 1
    
    # Check if edge already exists
    if check_edge(G_new.edge_index, torch.tensor([[start], [end]])):
        # If edge exists, return original G 
        return G, False, False
    else:
        # Add edge from start to end
        G_new.edge_index = append_edge(G_new.edge_index, torch.tensor([[start], [end]]))
        G_new.edge_index = append_edge(G_new.edge_index, torch.tensor([[end], [start]]))
        #update degree of start and end node
        G_new.x[start, 0] += 1
        G_new.x[end, 0] += 1
    
    return G_new, True, False

def sample_is_acyclic_gflownet(gflownet, G_t, max_actions, min_node_count):
  actions_taken = 0

  P_F_s, P_F_e, P_B_s, P_B_e = gflownet(G_t.x, G_t.edge_index)
  total_P_F = 0
  total_P_B = 0
  stop = False
  while actions_taken < max_actions:

    # Sample starting node from P_F_s
    cat_start = Categorical(logits=P_F_s)
    starting_node = cat_start.sample()

    # Mask out the starting node      
    mask = torch.ones_like(P_F_e)
    mask[starting_node] = 0
    
    P_F_e = P_F_e * mask - 100 * (1 - mask)

    # Sample ending node from P_F_e
    cat_end = Categorical(logits=P_F_e)
    ending_node = cat_end.sample()

    # Action
    action = (starting_node, ending_node)

    # If action is to stay in the same state, we don't need to create a new graph
    if action[0] == action[1]:
      G_new = G_t.clone()
      valid = False
    else:
      # Create new graph
      G_new, valid, stop = get_new_state_is_acyclic(G_t, action)
  
    actions_taken += 1

    if not valid:
      total_P_F += torch.tensor(0.0)
    else:
      total_P_F += cat_start.log_prob(action[0]) + cat_end.log_prob(action[1])
      
    # We recompute P_F and P_B for new_state
    P_F_s, P_F_e, P_B_s, P_B_e = gflownet(G_new.x, G_new.edge_index)

    # Accumulate the P_B sum 
    if not valid:
      total_P_B += torch.tensor(0.0)
    else:
      total_P_B += Categorical(logits=P_B_s).log_prob(action[0]) + Categorical(logits=P_B_e).log_prob(action[1])

    # # Continue iterating
    G_t = G_new.clone()

    if stop and G_t.x.size(0) >= min_node_count:
      break
    
  return G_t

def generate_graphs(gflownet, n, actions=10, min_nodes=5):
  graphs = []
  cyclic_count = 0
  acyclic_count = 0
  for i in range(n):
    G_t = create_initial_graph_is_acyclic(num_node_features=1)
    G = sample_is_acyclic_gflownet(gflownet, G_t, actions, min_nodes)
    G.num_nodes = G.x.size(0)
    cycles = nx.cycle_basis(to_networkx(G, to_undirected=True))
    if len(cycles) > 0:
      G.y = torch.tensor([0])
      cyclic_count += 1
    else:
      G.y = torch.tensor([1])
      acyclic_count += 1
    
    if G.x.size(0) > 1:
      graphs.append(G)
  print("Generated {} cyclic graphs and {} acyclic graphs".format(cyclic_count, acyclic_count))
  return graphs