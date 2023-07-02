import torch
import torch.nn as nn
from torch import unsqueeze
from Encoder import Encoder
from Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=512,
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0.2,
                 device="cuda",
                 max_length=100):
        """
        Transformer model constructor.

        Inputs:
        - src_vocab_size: Vocabulary size of the source language
        - trg_vocab_size: Vocabulary size of the target language
        - src_pad_idx: Padding index for source language
        - trg_pad_idx: Padding index for target language
        - embed_size: Dimensionality of the token embeddings (default: 512)
        - num_layers: Number of layers in the encoder and decoder (default: 6)
        - forward_expansion: Factor for increasing hidden dimension size in feed-forward networks (default: 4)
        - heads: Number of attention heads (default: 8)
        - dropout: Dropout rate (default: 0.2)
        - device: Device to run the model on (default: "cuda" if available, else "cpu")
        - max_length: Maximum sequence length (default: 100)
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size,
                               embed_size,
                               num_layers,
                               heads,
                               device,
                               forward_expansion,
                               dropout,
                               max_length
                               )
        self.decoder = Decoder(trg_vocab_size,
                               embed_size,
                               num_layers,
                               heads,
                               forward_expansion,
                               dropout,
                               device,
                               max_length
                               )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        """
        Create a mask for the source sequence.

        Inputs:
        - src: Source sequence tensor

        Returns:
        - src_mask: Source mask tensor
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        """
        Create a mask for the target sequence.

        Inputs:
        - trg: Target sequence tensor

        Returns:
        - trg_mask: Target mask tensor
        """
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).unsqueeze(0).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        """
        Forward pass of the Transformer model.

        Inputs:
        - src: Source sequence tensor
        - trg: Target sequence tensor

        Returns:
        - out: Output tensor from the Transformer model
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask)

        return out
