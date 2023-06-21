import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,output_size,num_layers,drop_out):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop_out)
        self.embedding = nn.Embedding(input_size,embedding_size)
        self.rnn = nn.LSTM(embedding_size,hidden_size, num_layers, dropout = drop_out)
        self.fc = nn.Linear(hidden_size,output_size)
    def forward(self, x, hidden, cell):
        #shape of x: (N)  but we want (1,N)
        x = x.unsqueeze(0)
        #embedding shape (1,N,embedding_size)
        embedding = self.dropout(self.embedding(x))
        outputs,(hidden,cell)=self.rnn(embedding,(hidden,cell))
        predictions = self.fc(outputs)
        #shape of predictions: (1, N,len_voc)
        predictions = predictions.squeeze(0)

        return  predictions, hidden,cell




