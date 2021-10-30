
from transformers import RobertaTokenizer, RobertaModel
from transformers.file_utils import TRANSFORMERS_CACHE
import torch
from torch import nn
import torch.nn.functional as F

from data import ParamTest

# ToDo: determine DROPOUT_RATE
DROPOUT_RATE = 0.5
# ToDo: determine CLASSIFICATION_HIDDEN_SIZE
CLASSIFICATION_HIDDEN_SIZE = 10
# ToDo: determine output_size
output_size = 3
# ToDo: determine NUM_CLASSES
NUM_CLASSES = 2

# ToDo: determine labels
labels = ""

# # ToDo: determine factors
# factor = None

class Network(nn.Module):

    def __init__(self,output_size):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(output_size,CLASSIFICATION_HIDDEN_SIZE)
        self.fc2 = nn.Linear(CLASSIFICATION_HIDDEN_SIZE,CLASSIFICATION_HIDDEN_SIZE)
        self.classification_dropout_layer = nn.Dropout(p=DROPOUT_RATE)
        self.output_layer = nn.Linear(CLASSIFICATION_HIDDEN_SIZE,NUM_CLASSES)

    def forward(self,input):
        output = self.fc1(input)
        output = F.relu(output)
        output = self.fc2(output)
        output = self.classification_dropout_layer(output)
        output = self.output_layer(output)
        return output

class CodeBERT:

    def __init__(self):
        self.model = RobertaModel.from_pretrained(
        "microsoft/codebert-base", cache_dir=TRANSFORMERS_CACHE)
        self.tokenizer = RobertaTokenizer.from_pretrained(
        "microsoft/codebert-base", cache_dir=TRANSFORMERS_CACHE)
        self.max_length = 512

    def get_inputs(self,input_text):
        tokens = self.tokenizer.tokenize(input_text)
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        padding_length = self.max_length - len(tokens)
        token_ids += [self.tokenizer.pad_token_id]*padding_length
        return token_ids, length

    def get_bow_representation(self,sequence):
        input_ids, length = self.get_inputs(sequence)
        mask = (torch.arange(self.max_length) < length).unsqueeze(-1).float()

        # embeddings = model.embeddings(input_ids) * mask
        embeddings = self.model.embeddings(torch.tensor(input_ids).unsqueeze(0)) * mask

        # factor ?
        # vector = torch.sum(embeddings, dim=1)/torch.sum(factor, dim=1)
        # return vector
        return torch.sum(embeddings, dim=1)


def baseline():
    cb = CodeBERT()

    comment_vector = cb.get_bow_representation(comment_sequence)

    code_vector = cb.get_bow_representation(code_sequence)
    
    feature_vector = torch.cat([comment_vector, code_vector], dim=-1)

    print(feature_vector)


    # logits = output_layer(classification_dropout_layer(
    #     F.relu(fc1(feature_vector))))

    # print(logits)

    # logprobs = F.log_softmax(logits, dim=-1)

    # print(logprobs)

    # loss = F.nll_loss(logprobs, labels)


if __name__ == "__main__":

    comment_sequence, code_sequence = ParamTest(
        "old_comment_raw", "old_code_raw")

    comment_sequence, code_sequence = comment_sequence[0], code_sequence[0]

    baseline()
