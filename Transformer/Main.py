import torch
from Transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input data
sequence_length = 100
embed_size = 512
src_vocab_size = 10000
trg_vocab_size = 10000
src_pad_idx = 0
trg_pad_idx = 0

# Generate random input sequence
input_sequence = torch.randint(low=1, high=src_vocab_size, size=(1, sequence_length)).to(device)
embed_size = int(embed_size)
# Initialize the Transformer model
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=embed_size).to(device)

# Pass the input sequence through the model
output = model(input_sequence, input_sequence)

print("Input shape:", input_sequence.shape)
print("Output shape:", output.shape)
