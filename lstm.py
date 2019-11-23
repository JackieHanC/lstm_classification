import torch
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 

class LSTM(nn.Module):
    def __init__(self, params):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(params["vocab_size"], \
            params["embedding_dim"])
        
        self.lstm = nn.LSTM(params["embedding_dim"], \
            params["lstm_hidden_dim"], batch_first=True)

        self.fc = nn.Linear(params["seq_max_length"] * params["lstm_hidden_dim"], params["num_of_tags"])

    def forward(self, s):
        # apply the embedding layer that maps each token to its embedding
        # dim: batch_size * batch_max_len * embedding_dim
        s = self.embedding(s)

        # dim: batch_size * batch_max_len * lstm_hidden_dim
        s, _ = self.lstm(s)

        s = s.reshape(-1, s.shape[1] * s.shape[2])

        s = self.fc(s)
        return F.log_softmax(s, dim=1)