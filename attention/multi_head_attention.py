import os
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.insert(0,path)
from constants import MULTI_HEADS,DROPOUT_RATE,HIDDEN_SIZE
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention

# computing multi-head attention between each hidden state of the comment encoder and the hidden states of the code encoder 


# attention_states = compute_attention_states(old_nl_hidden_states, old_nl_masks, old_nl_hidden_states, transformation_matrix = self.nl_attention_transform_matrix, multihead_attention = self.self_attention))
# attention_states = torch.cat([attention_states,
# computer_attention_states(code_hidden_states, code_maks, old_nl_hidden_states, transformation_matrix = self.sequence_attention_transform_matrix, multihead_attention=self.code_sequence_multihead_attention)

# old_nl_hidden_states, old_nl_final_state = self.nl_encoder.forward(
	# self.nl_encoder = Encoder(NL_EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE)
 

out_dim = 2 * HIDDEN_SIZE  
comment_multi = MultiHeadAttention(MULTI_HEADS, out_dim, dropout=DROPOUT_RATE)
code_multi = MultiHeadAttention(MULTI_HEADS, out_dim, dropout=DROPOUT_RATE)


def compute_attention_states(key_states, masks, query_states, transformation_matrix = None, multihead_attention = None):
	key = tf.einsum('bsh,hd->sbd',key_states,transformation_matrix)
	# ??? 
	query = query_states.permute(1,0,2)
	value = key
	attn_output,_ = multihead_attention(query,key,value, attention_mask = masks.squeeze(1))
	return attn_output.permute(1,0,2)

if __name__ == "__main__":
	# comment
	old_nl_hidden_states = None
	batch_data = None
	# old_nl_masks = (tf.reshape(tf.range(old_nl_hidden_states.shape[1])) >= batch_data.old_nl_lengths.view(-1, 1)).unsqueeze(1)
	
	nl_attention_transform_matrix = tf.Variable(tf.random.uniform([out_dim,out_dim],dtype=tf.dtypes.float32)) 	
	comment_attention_states = compute_attention_states(old_nl_hidden_states, old_nl_masks, old_nl_hidden_states, transformation_matrix = nl_attention_transform_matrix, multihead_attention = comment_multi))

	# code
	code_hidden_states = None
	code_masks = None				
	sequence_attention_transform_matrix = tf.Variable(tf.random.uniform([self.out_dim,self.out_dim],dtype=tf.dtypes.float32))
	code_attention_states = compute_attention_states(code_hidden_states, code_masks, old_nl_hidden_states, transformation_matrix = sequence_attention_transform_matrix, multihead_attention = code_multi))

	# final attention
	attention_states = tf.concat([comment_attention_states,code_attention_states])

