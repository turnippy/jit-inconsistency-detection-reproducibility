# BiGRU implementation in Tensorflow
# Contributors: DJ Jin, T Liu, J Yasmin
# CISC867 Reproducibility study 2021
# Queen's University, Canada

# The following is an implementation of a bidirectional GRU network.
# This implementation is based on the PyTorch equivalent implemented by
# S. Panthaplackel, J.J. Li, G. Milos, and R.J. Mooney (2020),
# hosted at https://github.com/panthap2/deep-jit-inconsistency-detection

import tensorflow as tf
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense

class BiGru (tf.Module):
    def __init__(self, embedding_size, hidden_size,
                 num_layers, dropout, bidirectional=True):
        super(BiGru, self).__init__()
        self.__rnn = Bidirectional(GRU(input_size=embedding_size,
                         hidden_size=hidden_size,
                         dropout=dropout,
                         num_layer=num_layers,
                         time_major=False)) #data on tensorflow is batch-major by default

    def forward(selfself, src_embedded_tokens, src_lengths, device):
        encoder_hidden_states, _ = self.__rnn.forward(src_embedded_tokens)
        return encoder_hidden_states, encoder_final_state