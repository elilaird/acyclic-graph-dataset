# acyclic-graph-dataset
Acyclic graph generation for evaluating graph neural networks

## Description
The script generates acyclic and cyclic graphs using various networkx functions. The networkx functions generate graphs up to the max number of nodes passed in as arguments. \
Pretrained GflowNets are additionally used to generate diverse acyclic and cyclic graphs to complement the networkx graphs. \
The script then converts the networkx graphs to a pytorch geometric graphs and saves them in the passed directory.

## Requirements
* Python >=3.6
* torch_geometric >=2.2.0
* torch >= 1.13.1
* networkx >= 2.8.4


## Example 
> These parameters for max_cyclic_nodes and max_acyclic_nodes and num_graphs generate 2405 graphs: 405 from networkx and 2000 from GflowNets. Note: networkx generation becomes very computationally expensive for max_cyclic_nodes > 70 and max_acyclic_nodes > 10.
```shell
python acyclic-graph-generation.py --max_cyclic_nodes 71 --max_acyclic_nodes 11 --num_gflownet_graphs 1000 --seed 1 --save_dir ./data          
```

