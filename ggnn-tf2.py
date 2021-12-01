import tensorflow as tf

import tf2_gnn as tfgnn

layer_input = tfgnn.layers.GNNInput(
    node_features=tf.random.normal(shape=(5, 3)),
    adjacency_lists=(
        tf.constant([[0, 1], [1, 2], [3, 4]], dtype=tf.int32),
        tf.constant([[1, 2], [3, 4]], dtype=tf.int32),
        tf.constant([[2, 0]], dtype=tf.int32)
    ),
    node_to_graph_map=tf.fill(dims=(5,), value=0),
    num_graphs=1,
)

params = tfgnn.layers.GNN.get_default_hyperparameters()
params["hidden_dim"] = 12
layer = tfgnn.layers.GNN(params)
output = layer(layer_input)
print(output)