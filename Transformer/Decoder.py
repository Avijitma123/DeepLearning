import torch.nn as nn
import torch
from SelfAttention import SelfAttention
from Transformer_Block import Transformer_Block
from torch import arange

"""
    Here I hvae implemented decoder block
"""
class DecoderBlock(nn.Module):
    """
            Decoder Block module of the Transformer model.

            Inputs:
            - embed_size: Dimensionality of the token embeddings
            - heads: Number of attention heads
            - forward_expansion: Factor for increasing hidden dimension size in feed-forward networks
            - dropout: Dropout rate
            - device: Device to run the model on
    """
    def __init__(self,embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self). __init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm = nn.LayerNorm(embed_size)
        self.device = device
        self.transformer_block = Transformer_Block(
            embed_size, heads, dropout, forward_expansion,device
        )

        self.dropout = nn.Dropout(dropout)
    def forward(self, x, value, key, src_mask, trg_mask):
        """
               Forward pass of the Decoder Block module.

               Inputs:
               - x: Input tensor representing the decoder input
               - value: Value tensor from the encoder output
               - key: Key tensor from the encoder output
               - src_mask: Mask tensor for the source sequence
               - trg_mask: Mask tensor for the target sequence

               Returns:
               - out: Output tensor of the decoder block
        """
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value,key,query,src_mask)
        return out


class Decoder(nn.Module):
    """
            Decoder module of the Transformer model.

            Inputs:
            - trg_vocab_size: Vocabulary size of the target language
            - embed_size: Dimensionality of the token embeddings
            - num_layers: Number of layers in the decoder
            - heads: Number of attention heads
            - forward_expansion: Factor for increasing hidden dimension size in feed-forward networks
            - dropout: Dropout rate
            - device: Device to run the model on
            - max_length: Maximum sequence length
    """
    def __init__(self,
                 trg_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length):
        super(Decoder,self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size,embed_size)
        self.position_embedding = nn.Embedding(max_length,embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size,heads,forward_expansion,dropout,device) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size,trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask, trg_mask):
        """
                Forward pass of the Decoder module.

                Inputs:
                - x: Input tensor representing the target sequence
                - enc_out: Encoder output tensor
                - src_mask: Mask tensor for the source sequence
                - trg_mask: Mask tensor for the target sequence

                Returns:
                - out: Output tensor of the decoder
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N,seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x))+ self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x,enc_out, enc_out,src_mask,trg_mask)
        out = self.fc_out(x)
        return out


