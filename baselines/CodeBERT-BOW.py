
from transformers import RobertaTokenizer, RobertaModel
from transformers.file_utils import TRANSFORMERS_CACHE
import torch
from torch import nn


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
    mask = (torch.arange(max_length) < length).unsqueeze(-1).float()
    embeddings = model.embeddings(input_ids) * mask
    # factor ?
    vector = torch.sum(embeddings, dim=1)/torch.sum(factor, dim=1)
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

    classification_dropout_layer = nn.Dropout(p=DROPOUT_RATE)

    fc1 = nn.Linear(output_size, CLASSIFICATION_HIDDEN_SIZE)
    fc2 = nn.Linear(CLASSIFICATION_HIDDEN_SIZE, CLASSIFICATION_HIDDEN_SIZE)
    output_layer = nn.Linear(CLASSIFICATION_HIDDEN_SIZE, NUM_CLASSES)

    comment_vector = get_bow_representation(comment_sequence)

    code_vector = get_bow_representation(code_sequence)

    feature_vector = torch.cat([comment_vector, code_vector], dim=-1)

    logits = output_layer(classification_dropout_layer(
        torch.nn.functional.relu(fc1(feature_vector))))

    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    loss = torch.nn.functional.nll_loss(logprobs, labels)
