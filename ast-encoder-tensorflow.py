# AST encoder implementation in tf
# Contributors: DJ Jin, T Liu, J Yasmin
# CISC867 Reproducibility study 2021
# Queen's University, Canada

import os
import tensorflow as tf

from constants import *
from ggnn-tensorflow import GatedGraphNeuralNetwork, AdjacencyList

class ASTGraphEncoder(tf.Module):
    """Encoder which learns a representation of a method's AST. The underlying network is a Gated Graph Neural Network."""
    def __init__(self, hidden_size, num_edge_types):
        super(ASTGraphEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_edge_types = num_edge_types
        self.gnn = GatedGraphNeuralNetwork(self.hidden_size, self.num_edge_types,
            [GNN_LAYER_TIMESTEPS], {}, GNN_DROPOUT_RATE, GNN_DROPOUT_RATE)
    
    def __call__(self, initial_node_representation, graph_batch, device):
        adjacency_lists = []
        for edge_type in range(self.num_edge_types):
            adjacency_lists.append(AdjacencyList(node_num=graph_batch.num_nodes,
                adj_list=graph_batch.edges[edge_type], device=device))
        node_representations = self.gnn.compute_node_representations(
            initial_node_representation=initial_node_representation, adjacency_lists=adjacency_lists)
        hidden_states = node_representations[graph_batch.node_positions]
        return hidden_states