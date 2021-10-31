
from transformers import RobertaTokenizer, RobertaModel
from transformers.file_utils import TRANSFORMERS_CACHE
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random

from data import ParamDataSet

# ToDo: determine DROPOUT_RATE
DROPOUT_RATE = 0.5
# ToDo: determine CLASSIFICATION_HIDDEN_SIZE
CLASSIFICATION_HIDDEN_SIZE = 10
# ToDo: determine output_size
output_size = 3
# ToDo: determine NUM_CLASSES
NUM_CLASSES = 1

# # ToDo: determine factors
# factor = None

class Network(nn.Module):

    def __init__(self,input_size):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(input_size,CLASSIFICATION_HIDDEN_SIZE)
        self.fc2 = nn.Linear(CLASSIFICATION_HIDDEN_SIZE,CLASSIFICATION_HIDDEN_SIZE)
        # self.classification_dropout_layer = nn.Dropout(p=DROPOUT_RATE)
        self.output_layer = nn.Linear(CLASSIFICATION_HIDDEN_SIZE,NUM_CLASSES)

    def forward(self,input):
        output = self.fc1(input)
        output = F.relu(output)
        output = self.fc2(output)
        # output = self.classification_dropout_layer(output)
        output = self.output_layer(output)
        # output = F.softmax(output,dim=-1)
        output = F.sigmoid(output)
        return output

    def train(self,dataset):
        iter = 0
        for data,labels in dataset:

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(data)

            # Calculate Loss: softmax --> cross entropy loss
            loss = F.nll_loss(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            print(f'{iter} {loss.item()}')
        print(f'{iter} {loss.item()}')


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

class DataSet:

    def __init__(self):
        self.cb = CodeBERT()
        self.max_length = self.cb.max_length

    def get_raw_data(self,dataset):
        return ParamDataSet(dataset,
        "old_comment_raw", "old_code_raw", "label")

    def get_feature_vector(self,comment_sequence,code_sequence):
        comment_vector = self.cb.get_bow_representation(comment_sequence)
        code_vector = self.cb.get_bow_representation(code_sequence)
        feature_vector = torch.cat([comment_vector, code_vector], dim=-1)
        return feature_vector

    def get_data_set(self,dataset):
        comment_sequences, code_sequences, labels = self.get_raw_data(dataset)
        print(len(comment_sequences))
        test_set = list()
        for comment_sequence,code_sequence,label in tqdm(zip(comment_sequences,code_sequences,labels)):
            feature_vector = self.get_feature_vector(comment_sequence,code_sequence)
            test_set.append([feature_vector,torch.tensor([label])])
    
        return test_set

if __name__ == "__main__":
    print(f"CodeBERT-BOW Baseline")

    ds = DataSet()

    # train data set require memory more than 32 GB
    data_set = ds.get_data_set('test')
    train,test = data_set[:int(len(data_set)*0.8)],data_set[int(len(data_set)*0.8):]
    print(f'{len(train)} {len(test)}')

    # forward network
    input_size = 1536
    model = Network(input_size)
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

    # train the model
    # enable to run, but not sure
    model.train(train)

    test_output = model(test[0][0])

    print(test_output)
    print(test[0])


    
