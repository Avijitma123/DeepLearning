import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()

        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        x = torch.matmul(attention, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x = self.out_linear(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = Attention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))

        mlp_output = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_output))

        return x

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, num_classes, hidden_dim=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.patch_embedding = PatchEmbedding(d_model, patch_size=16)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        x = self.patch_embedding(x)
        batch_size, seq_len, _ = x.size()

        for layer in self.encoder_layers:
            x = layer(x, mask)

        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)

        return x
