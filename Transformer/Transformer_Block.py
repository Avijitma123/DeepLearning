import torch
from SelfAttention import SelfAttention
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __int__(self,embed_size,heads,dropout,forward_expansion):
        super(TransformerBlock,self).__int__()
        self.attention = SelfAttention(embed_size=embed_size,heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        #FFN
        self.feed_forward = nn.Sequential(
             nn.Linear(embed_size,forward_expansion*embed_size),
             nn.ReLU(),
             nn.Linear(forward_expansion*embed_size,embed_size)

         )
        self.dropout = nn.Dropout(dropout)

    def forward(self,value,key,quary,mask):
        attention = self.attention(value,key,quary,mask)
        x=self.dropout(self.norm1(attention+quary))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        return out




