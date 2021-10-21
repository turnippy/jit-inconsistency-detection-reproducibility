# GGNN implementation in Tensorflow
# Contributors: DJ Jin, T Liu, J Yasmin
# CISC867 Reproducibility study 2021
# Queen's University, Canada

# The following is an implementation of a gate graph neural network
# proposed by Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel (2016).
# The implementation is based on the PyTorch equivalent implemented by
# S. Panthaplackel, J.J. Li, G. Milos, and R.J. Mooney (2020),
# hosted at https://github.com/panthap2/deep-jit-inconsistency-detection

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils
import tensorflow.keras.layers
from typing import List, Tuple, Dict, Sequence, Any

class AdjList:
    # topological representation of graph

    # params: num_nodes:int, adj_list: List, device: tf.device
    def __init__(self, num_nodes, adj_list, device):
        self.num_nodes = num_nodes
        self.adj_list = adj_list
        self.data = tf.tensor(adj_list, dtype=tf.float64, device=device)
        self.edge_num = len(adj_list)

    def device(self):
        return self.data.device

    def __getitem__(selfself, item):
        return self.data[item]

class ggnn(tf.module):
    # default constructor values are taken from Panthaplackel et al. (2020)
    def __init__(self, hidden_size, num_edge_types, layer_timesteps,
                 residual_connections,
                 state_to_message_dropout=0.3,
                 rnn_dropout=0.3,
                 use_bias_for_message_linear=True):

        super(ggnn, self).__init__()

        self.hidden_size = hidden_size
        self.num_edge_types = num_edge_types
        self.layer_timesteps = layer_timesteps
        self.residual_connections = residual_connections
        self.state_to_message_dropout = state_to_message_dropout
        self.rnn_dropout = rnn_dropout
        self.use_bias_for_message_linear = use_bias_for_message_linear

        # configure rnn cells in each layer
        # encode node states as messages
        self.state_to_message_linears = []
        self.rnn_cells = []

        # iterate through layers by timestep
        for layer_idx in range(len(self.layer_timesteps)):
            # set of transformations for node state to messages for the current layer
            state_to_msg_linears_cur_layer = []
            # iterate through all edge types in graph and encode into appropriate message
            for edge_type_j in range(self.num_edge_types):
                # apply a linear transformation (tf Dense layer is equivalent of torch.linear)
                state_to_msg_linear_layer_i_type_j = tf.keras.layers.Dense(self.hidden_size, self.hidden_size,
                                                                           use_bias=use_bias_for_message_linear,
                                                                           activation=None)
                setattr(self,
                        'state_to_message_linear_layer%d_type%d,' % (layer_idx, edge_type_j),
                        state_to_msg_linear_layer_i_type_j
                        )
                state_to_msg_linears_cur_layer.append(state_to_msg_linear_layer_i_type_j)
                # collected all the encoded states into the state_to_message_linears attr
            self.state_to_message_linears.append(state_to_msg_linears_cur_layer)

            layer_residual_connections = self.residual_connections.get(layer_idx, [])
            rnn_cell_layer_i = tf.keras.layers.GRUCell(self.hidden_size * (1 + len(layer_residual_connections)),
                                                       self.hidden_size)
            setattr(self, 'rnn_cell_layer%d' % layer_idx, rnn_cell_layer_i)
            self.rnn_cells.append(rnn_cell_layer_i)

        # set dropout values (if applicable) for the ggnn
        self.state_to_message_dropout_layer = tf.keras.layers.Dropout(self.state_to_message_dropout)
        self.rnn_dropout_layer = tf.keras.layers.Dropout(self.rnn_dropout)

    def device(self):
        # not sure how to get hidden hidden tensor in tf, or what is equivalent
        # see issues
        # return self.rnn_cells[0]
        pass

    def forward(self, initial_node_representation, adj_lists, return_all_states=False):
        return self.compute_node_representation(initial_node_representation, adj_lists, return_all_states)

    def compute_node_representation(self,
                                    initial_node_representation,
                                    adj_lists,
                                    return_all_states=False):
        init_node_repr_size = initial_node_representation.size(1)
        device = adj_lists[0].data.device
        if init_node_repr_size < self.hidden_size:
            # if dimensions of embedding are smaller than current node dimension, pad embedding
            pad_size = self.hidden_size - init_node_repr_size
            zeroes = 