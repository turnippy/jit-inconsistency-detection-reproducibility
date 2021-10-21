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

    # params: node_num:int, adj_list: List, device: tf.device
    def __init__(self, node_num, adj_list, device):
        self.node_num = node_num
        self.adj_list = adj_list
        self.data = tf.Tensor(adj_list, dtype=tf.float64)
        self.data.device = device
        self.edge_num = len(adj_list)

    def device(self):
        return self.data.device

    def __getitem__(self, item):
        return self.data[item]

class GGNN(tf.Module):
    # default constructor values are taken from Panthaplackel et al. (2020)
    def __init__(self, hidden_size, num_edge_types, layer_timesteps,
                 res_conns,
                 state_to_msg_dropout=0.3,
                 rnn_dropout=0.3,
                 use_bias_for_msg_lin=True):

        super(GGNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_edge_types = num_edge_types
        self.layer_timesteps = layer_timesteps
        self.res_conns = res_conns
        self.state_to_msg_dropout = state_to_msg_dropout
        self.rnn_dropout = rnn_dropout
        self.use_bias_for_msg_lin = use_bias_for_msg_lin

        # configure rnn cells in each layer
        # encode node states as messages
        self.state_to_msg_lins = []
        self.rnn_cells = []

        # iterate through layers by timestep
        for layer_idx in range(len(self.layer_timesteps)):
            # set of transformations for node state to messages for the current layer
            state_to_msg_lins_cur_layer = []
            # iterate through all edge types in graph and encode into appropriate message
            for edge_type_j in range(self.num_edge_types):
                # apply a linear transformation (tf Dense layer is equivalent of torch.linear)
                state_to_msg_lin_layer_i_type_j = tf.keras.layers.Dense(self.hidden_size, self.hidden_size,
                                                                           use_bias=use_bias_for_msg_lin,
                                                                           activation=None)
                setattr(self,
                        'state_to_msg_lin_layer%d_type%d,' % (layer_idx, edge_type_j),
                        state_to_msg_lin_layer_i_type_j
                        )
                state_to_msg_lins_cur_layer.append(state_to_msg_lin_layer_i_type_j)
                # collected all the encoded states into the state_to_message_linears attr
            self.state_to_msg_lins.append(state_to_msg_lins_cur_layer)

            layer_res_conns = self.res_conns.get(layer_idx, [])
            rnn_cell_layer_i = tf.keras.layers.GRUCell(self.hidden_size * (1 + len(layer_res_conns)),
                                                       self.hidden_size)
            setattr(self, 'rnn_cell_layer%d' % layer_idx, rnn_cell_layer_i)
            self.rnn_cells.append(rnn_cell_layer_i)

        # set dropout values (if applicable) for the ggnn
        self.state_to_msg_dropout_layer = tf.keras.layers.Dropout(self.state_to_msg_dropout)
        self.rnn_dropout_layer = tf.keras.layers.Dropout(self.rnn_dropout)

    def device(self):
        # not sure how to get hidden hidden tensor in tf, or what is equivalent
        # see issues
        # return self.rnn_cells[0]
        pass

    def forward(self, init_node_representation, adj_lists, return_all_states=False):
        return self.compute_node_representation(init_node_representation, adj_lists, return_all_states)

    def compute_node_representation(self,
                                    init_node_representation,
                                    adj_lists,
                                    return_all_states=False):
        init_node_repr_size = init_node_representation.size(1)
        device = adj_lists[0].data.device
        if init_node_repr_size < self.hidden_size:
            # if dimensions of embedding are smaller than current node dimension, pad embedding
            pad_size = self.hidden_size - init_node_repr_size
            zeroes = tf.zeros(init_node_representation.size(0), pad_size, dtype=tf.float32, device=device)
            init_node_representation = tf.concat([init_node_representation, zeroes], concat_dim=-1)
        node_states_per_layer = [init_node_representation]

        node_num = init_node_representation.size(0)

        msg_targets = []
        for edge_type_idx, adj_list_for_edge_type in enumerate(adj_lists):
            if adj_list_for_edge_type > 0: # see graph encoding for edge type info
                edge_targets = adj_list_for_edge_type[:, 1]
                msg_targets.append(edge_targets)
        msg_targets = tf.concat(msg_targets, concat_dim=0)

        for layer_idx, num_timesteps in enumerate(self.layer_timesteps):
            # Used shape abbreviations:
            #   V ~ number of nodes
            #   D ~ state dimension
            #   E ~ number of edges of current type
            #   M ~ number of messages (sum of all E)
            layer_res_conns = self.res_conns.get(layer_idx, [])
            layer_res_states = [node_states_per_layer[res_layer_idx]
                                     for res_layer_idx in layer_res_conns]
            node_states_curr_layer = node_states_per_layer[-1]
            for t in range(num_timesteps):
                msgs = []
                msg_source_states = []

                for edge_type_idx, adj_list_for_edge_type in enumerate(adj_lists):
                    if adj_list_for_edge_type.edge_num > 0:
                        edge_sources = adj_list_for_edge_type[:, 0]
                        edge_source_states = node_states_curr_layer[edge_sources]
                        f_state_to_msg = self.state_to_msg_lins[layer_idx][edge_type_idx]
                        all_msgs_for_edge_type = self.state_to_msg_dropout_layer(f_state_to_msg(edge_source_states))

                        msgs.append(all_msgs_for_edge_type)
                        msg_source_states.append(edge_source_states)

                msgs = tf.concat(msgs, concat_dim=0)

                inc_msgs = tf.zeros(node_num, msgs.size(1), device=device)
                # must find equivalent operation for scatter_add_ in tf
                #inc_msgs = tf.tensor_scatter_nd_add(inc_msgs,

                inc_info = tf.concat(layer_res_states + [inc_msgs], concat_dim=-1)

                updated_node_states = self.rnn_cells[layer_idx](inc_info, node_states_curr_layer)
                updated_node_states = self.rnn_dropout_layer(updated_node_states)

            node_states_per_layer.append(node_states_curr_layer)

        if return_all_states:
            return node_states_per_layer[1:]

        else:
            return node_states_per_layer[-1]

def main():
    ggnn_instance = GGNN(hidden_size=64, num_edge_types=2, layer_timesteps=[3,5,7,2],
                         res_conns={2: [0], 3: [0,1]})
    adj_list_type1 = AdjList(node_num=4, adj_list=[(0, 2), (2, 1), (1, 3)], device=ggnn_instance.device)
    adj_list_type2 = AdjList(node_num=4, adj_list=[(0, 0), (0, 1)], device=ggnn_instance.device)

    init_tensor = tf.ramdom.uniform(shape=[2,64])
    node_representations = ggnn_instance.compute_node_representations(init_node_representations=init_tensor,
                                                                      adj_lists=[adj_list_type1, adj_list_type2])

    print(node_representations)

if __name__ == '__main__':
    main()