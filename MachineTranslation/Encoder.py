import torch.nn as nn


class Encoder(nn.Module):
    def __init__ (self, input_size, embedding_size, hidden_size,num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size,embedding_size)
        self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,dropout=dropout)
    def forward(self,x):
        #x_shape: (seq_length, N)
        embedding = self.dropout(self.embedding(x))

        output,(hidden,cell) = self.rnn(embedding)
        return hidden, cell


