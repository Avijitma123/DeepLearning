import torch
import torch.nn as nn
from Transformer_Block import Transformer_Block

class Encoder(nn.Module):
    def __init__(self, source_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        """
        Encoder module of the Transformer model.

        Inputs:
        - source_vocab_size: Vocabulary size of the source language
        - embed_size: Dimensionality of the token embeddings
        - num_layers: Number of layers in the encoder
        - heads: Number of attention heads
        - device: Device to run the model on
        - forward_expansion: Factor for increasing hidden dimension size in feed-forward networks
        - dropout: Dropout rate
        - max_length: Maximum sequence length
        """
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(source_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                Transformer_Block(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    device=device
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Forward pass of the Encoder module.

        Inputs:
        - x: Input tensor representing the source sequence
        - mask: Mask tensor for source sequence

        Returns:
        - out: Encoded tensor representing the source sequence
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out
