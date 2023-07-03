import torch
import torch.nn as nn
import torch.nn.functional as F
"""
The PatchEmbedding class represents the patch embedding layer of the transformer model. 
It takes as input an image tensor and applies a convolutional operation to extract patches
from the image. These patches are then flattened and transposed to obtain the embeddings. 
The resulting tensor has the shape (batch_size, num_patches, d_model), where batch_size is 
the number of images in the batch, num_patches is the total number of patches in the image, 
and d_model is the dimension of the embedding.

"""
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        # Convolutional layer to extract patches and convert them to embeddings

    def forward(self, x):
        x = self.conv(x)  # Apply convolution to input image
        x = x.flatten(2).transpose(1, 2)
        # Flatten the spatial dimensions (height and width) of each patch
        # Transpose the tensor to have the embedding dimensions as the second dimension
        return x


"""
The MLP class represents the multi-layer perceptron (MLP) component of the transformer model. 
It takes as input an embedding tensor and applies two fully connected layers with a GELU activation 
function in between. The purpose of the MLP is to introduce non-linearity and enable the model to 
learn complex patterns in the data. Dropout is applied to the intermediate representation for regularization.
"""
class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        # First fully connected layer to transform the input
        self.fc2 = nn.Linear(hidden_dim, d_model)
        # Second fully connected layer to transform the intermediate representation back to the original dimension
        self.dropout = nn.Dropout(dropout)
        # Dropout layer for regularization

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        # Apply the first fully connected layer and pass it through the GELU activation function
        x = self.dropout(x)
        # Apply dropout to the output of the first layer
        x = self.fc2(x)
        # Apply the second fully connected layer
        x = self.dropout(x)
        # Apply dropout to the output of the second layer
        return x
"""
The Attention class represents the attention mechanism used in the transformer model. 
It takes as input query, key, and value tensors and performs scaled dot-product attention. 
The attention scores are calculated by taking the dot product between the query and key, 
divided by the square root of the head dimension. Softmax activation is applied to obtain the 
attention weights, and dropout is applied to the attention weights for regularization. 
The values are then weighted by the attention weights and projected using the output linear layer.
"""
class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Calculate the dimension of each head

        self.q_linear = nn.Linear(d_model, d_model)
        # Linear layer for the query projection
        self.k_linear = nn.Linear(d_model, d_model)
        # Linear layer for the key projection
        self.v_linear = nn.Linear(d_model, d_model)
        # Linear layer for the value projection

        self.dropout = nn.Dropout(dropout)
        # Dropout layer for regularization
        self.out_linear = nn.Linear(d_model, d_model)
        # Linear layer for the output projection

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()

        query = self.q_linear(query)
        # Project the query
        key = self.k_linear(key)
        # Project the key
        value = self.v_linear(value)
        # Project the value

        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Reshape and transpose the query tensor to facilitate multi-head attention
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Reshape and transpose the key tensor to facilitate multi-head attention
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Reshape and transpose the value tensor to facilitate multi-head attention

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # Calculate the attention scores using the dot product between query and key
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # Apply mask to attention scores if provided

        attention = F.softmax(scores, dim=-1)
        # Apply softmax activation function to obtain attention weights
        attention = self.dropout(attention)
        # Apply dropout to attention weights

        x = torch.matmul(attention, value)
        # Calculate the weighted sum of values using the attention weights
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # Reshape and transpose the output tensor
        x = self.out_linear(x)
        # Project the output tensor

        return x
"""
The EncoderLayer class represents a single layer in the encoder of the transformer model. 
It consists of two sub-layers: self-attention and a feed-forward neural network (MLP). 
Layer normalization is applied before and after each sub-layer. The input tensor is passed 
through the self-attention mechanism, and its output is added to the input tensor with a 
residual connection. Layer normalization is applied to the sum, and the result is passed 
through the feed-forward neural network. Again, the output of the MLP is added to the 
input tensor with a residual connection and layer normalization is applied.
"""
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = Attention(d_model, n_heads, dropout)
        # Self-attention mechanism
        self.norm1 = nn.LayerNorm(d_model)
        # Layer normalization after the first sublayer
        self.mlp = MLP(d_model, hidden_dim, dropout)
        # Feed-forward neural network (MLP) as the second sublayer
        self.norm2 = nn.LayerNorm(d_model)
        # Layer normalization after the second sublayer
        self.dropout = nn.Dropout(dropout)
        # Dropout layer for regularization

    def forward(self, x, mask=None):
        attention_output = self.attention(x, x, x, mask)
        # Apply self-attention mechanism to the input tensor
        x = self.norm1(x + self.dropout(attention_output))
        # Add the residual connection and apply layer normalization

        mlp_output = self.mlp(x)
        # Apply the feed-forward neural network to the output of the first sublayer
        x = self.norm2(x + self.dropout(mlp_output))
        # Add the residual connection and apply layer normalization

        return x


"""
The Transformer class represents the complete transformer model. 
It consists of a patch embedding layer to convert the input images 
into patches, a stack of encoder layers,layer normalization, and a
fully connected layer for classification.In the forward method, 
the input images are first passed through the patch embedding layer 
to obtain the patch representations. Then, the patch representations 
are passed through each encoder layer in the stack. After the encoder 
layers, layer normalization is applied to the output.Global average 
pooling is then performed to reduce the sequence length to 1, and finally, 
the output is passed through the fully connected layer for classification.
"""

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, num_classes, hidden_dim=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.patch_embedding = PatchEmbedding(d_model, patch_size=16)
        # Patch embedding layer to convert input images into patches

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ])
        # Stack of encoder layers

        self.norm = nn.LayerNorm(d_model)
        # Layer normalization after the encoder layers

        self.fc = nn.Linear(d_model, num_classes)
        # Fully connected layer for classification

    def forward(self, x, mask=None):
        x = self.patch_embedding(x)
        # Convert input images into patches

        batch_size, seq_len, _ = x.size()

        for layer in self.encoder_layers:
            x = layer(x, mask)
        # Pass the input through each encoder layer

        x = self.norm(x)
        # Apply layer normalization to the output of the encoder layers

        x = x.mean(dim=1)  # Global average pooling
        # Apply global average pooling to reduce the sequence length to 1

        x = self.fc(x)
        # Apply the fully connected layer for classification

        return x
