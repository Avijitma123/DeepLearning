# import torch
# from SelfAttention import SelfAttention
# import torch.nn as nn
#
#
# class Transformer_Block(nn.Module):
#     def __init__(self, embed_size, heads, dropout, forward_expansion):
#         super(Transformer_Block, self).__init__()
#         self.attention = SelfAttention(embed_size, heads)
#         self.norm1 = nn.LayerNorm(embed_size)
#         self.norm2 = nn.LayerNorm(embed_size)
#
#         # FFN
#         self.feed_forward = nn.Sequential(
#             nn.Linear(embed_size, forward_expansion * embed_size),
#             nn.ReLU(),
#             nn.Linear(forward_expansion * embed_size, embed_size)
#         )
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, value, key, query, mask):
#         attention = self.attention(value, key, query, mask)
#         x = self.dropout(self.norm1(attention + query))
#         forward = self.feed_forward(x)
#         out = self.dropout(self.norm2(forward + x))
#         return out


import torch
from SelfAttention import SelfAttention
import torch.nn as nn


class Transformer_Block(nn.Module):

    """Constructor"""
    def __init__(self, embed_size, heads, dropout, forward_expansion, device):
        super(Transformer_Block, self).__init__()
        """Create an object of SelfAttention class"""
        self.attention = SelfAttention(embed_size, heads)
        """set the device"""
        self.device = device
        """Layer Normalization"""
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        embed_size = int(embed_size)

        """# Feed Forward NN"""
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, int(forward_expansion * embed_size)),
            nn.ReLU(),
            nn.Linear(int(forward_expansion * embed_size), embed_size)
        )
        """Dropout layer"""
        self.dropout = nn.Dropout(float(dropout))




    """
    
           Forward method
    

   +----------------------------------+
   |            Input               |
   +----------------------------------+
                |
                v
   +----------------------------------+
   |       Self-Attention       |
   +----------------------------------+
                |
                v
   +----------------------------------+
   | Residual Connection & Layer Norm |
   +----------------------------------+
                |
                v
   +----------------------------------+
   |   Feed-Forward Neural Network   |
   +----------------------------------+
                |
                v
   +----------------------------------+
   | Residual Connection & Layer Norm |
   +----------------------------------+
                |
                v
   +----------------------------------+
   |            Output              |
   +----------------------------------+
    
    
    
    """

    def forward(self, value, key, query, mask):
        # Perform self-attention operation
        attention = self.attention(value, key, query, mask)

        # Apply dropout to the output of self-attention and add it to the original query
        x = self.dropout(self.norm1(attention + query))

        # Pass the output through the feed-forward neural network
        forward = self.feed_forward(x)

        # Apply dropout to the output of the feed-forward network and add it to the previous output
        out = self.dropout(self.norm2(forward + x))

        return out

