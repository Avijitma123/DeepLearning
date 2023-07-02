import torch
import torch.nn as nn


class SelfAttention(nn.Module):
	def __init__(self, embed_size, heads):
		super(SelfAttention, self).__init__()
		self.embed_size = embed_size
		self.heads = heads
		self.head_dim = embed_size // heads
		assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

		# Linear layers for values, keys, and queries
		self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

		# Linear layer for the final output
		self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

	def forward(self, values, keys, query, mask):
		print("===Inside forward===")
		N = query.shape[0]

		value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

		# Split embedding into self.heads pices
		values = values.reshape(N, value_len, self.heads, self.head_dim)
		keys = keys.reshape(N, key_len, self.heads, self.head_dim)
		queries = query.reshape(N, query_len, self.heads, self.head_dim)
		values = self.values(values)
		keys = self.keys(keys)
		queries = self.queries(queries)
		# Compute energy using matrix multiplication (einsum)
		energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
		# queries shape: (N, query_len, heads, heads_dim)
		# keys shape: ( N, key_len, heads, heads_dim)
		# energy shape: (N, heads, query_len, key_len)

		if mask is not None:
			# Apply mask to energy tensor if provided
			energy = energy.masked_fill(mask == 0, float("-1e20"))
		attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

		# Compute the attended values using matrix multiplication (einsum)
		out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
		out = self.fc_out(out)

		return out
