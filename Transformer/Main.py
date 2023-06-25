from SelfAttention import  SelfAttention
import torch

# Word embeddings for the sequence of words
word_embeddings = torch.randn(8, 32)

# Create an instance of the SelfAttention module
self_attention = SelfAttention(embed_size=32, heads=4)

# Forward pass through the self-attention module
output = self_attention(word_embeddings, word_embeddings, word_embeddings,1)

# Print the output shape
print(output.shape)