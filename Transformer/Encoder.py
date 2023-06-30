import torch
import torch.nn as nn
from Transformer_Block import Transformer_Block

#
# class Encoder(nn.Module):
#     def __init__(self,source_vocab_size,embed_size,num_layers,heads,device,forward_expansion,dropout,max_length):
#         super(Encoder,self).__init__()
#         self.embed_size = embed_size
#         self.device = device
#         self.word_embedding = nn.Embedding(source_vocab_size,embed_size)
#         self.position_embedding = nn.Embedding(max_length,embed_size)
#
#         self.layers = nn.ModuleList(
#             [
#                 Transformer_Block(
#                     embed_size,
#                     heads,
#                     dropout,
#                     forward_expansion
#                 )
#                 for _ in range(num_layers)
#             ]
#         )
#         self.dropout = dropout
#
#     def forward(self,x,mask):
#         N, seq_length = x.shape
#         positions = torch.arange(0, seq_length).expand(N,seq_length).to(self.device)
#         out=self.dropout(self.word_embedding(x) + self.position_embedding(positions))
#         for layer in self.layers:
#             out = layer(out,out,out,mask)
#         return out

class Encoder(nn.Module):
    def __init__(self, source_vocab_size, embed_size, num_layers, heads, device,
                 forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(source_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            Transformer_Block(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)  # Corrected assignment

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out