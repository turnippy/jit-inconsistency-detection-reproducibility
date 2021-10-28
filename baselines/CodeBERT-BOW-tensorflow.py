# CodeBERT BoW implementation in Tensorflow
# Contributors: DJ Jin, T Liu, J Yasmin
# CISC867 Reproducibility study 2021
# Queen's University, Canada

# The following is the same implementation of the CodeBERT
# Bag of Words baseline except using Tensorflow and Keras
# APIs
from transformers import RobertaTokenizer, RobertaModel
from transformers.file_utils import TRANSFORMERS_CACHE

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


# ToDo: determine DROPOUT_RATE
DROPOUT_RATE = 0.5
# ToDo: determine CLASSIFICATION_HIDDEN_SIZE
CLASSIFICATION_HIDDEN_SIZE = 10
# ToDo: determine output_size
output_size = 3
# ToDo: determine NUM_CLASSES
NUM_CLASSES = 4

# ToDo: determine comment_sequence
comment_sequence = ""
# ToDo: determine code_sequence
code_sequence = ""
# ToDo: determine labels
labels = ""
# ToDo: determine factors
factor = None


def get_inputs(input_text):

    tokens = tokenizer.tokenize(input_text)

    length = min(len(tokens), max_length)

    tokens = tokens[:length]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    padding_length = max_length - len(tokens)
    token_ids += [tokenizer.pad_token_id]*padding_length
    return token_ids, length


def get_bow_representation(sequence):

    input_ids, length = get_inputs(sequence, max_length)
    mask = tf.cast((tf.experimental.numpy.arange(max_length) < length).expand_dims(-1), dtype=tf.float32)
    embeddings = model.embeddings(input_ids) * mask
    # factor ?
    vector = tf.math.reduce_sum(embeddings, axis=1)/tf.math.reduce_sum(factor, axis=1)
    return vector


if __name__ == "__main__":
    print("CodeBERT-BOW baseline")

    tokenizer = RobertaTokenizer.from_pretrained(
        "microsoft/codebert-base", cache_dir=TRANSFORMERS_CACHE)
    model = RobertaModel.from_pretrained(
        "microsoft/codebert-base", cache_dir=TRANSFORMERS_CACHE)

    max_length = 512

    print(tokenizer)

    print(model)

    classification_dropout_layer = Dropout(rate=DROPOUT_RATE)

    fc1 = Dense(CLASSIFICATION_HIDDEN_SIZE, input_shape=(output_size,), activation=None)
    fc2 = Dense(CLASSIFICATION_HIDDEN_SIZE, input_shape=(CLASSIFICATION_HIDDEN_SIZE,), activation=None)
    output_layer = Dense(NUM_CLASSES, input_shape=(output_size,), activation=None)

    comment_vector = get_bow_representation(comment_sequence)

    code_vector = get_bow_representation(code_sequence)

    feature_vector = tf.concat([comment_vector, code_vector], axis=-1)

    logits = output_layer(classification_dropout_layer(
        torch.nn.functional.relu(fc1(feature_vector))))

    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    loss = torch.nn.functional.nll_loss(logprobs, labels)
