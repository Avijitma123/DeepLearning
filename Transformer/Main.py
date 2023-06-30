import torch
from SelfAttention import SelfAttention
from Transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.tensor([[1,5,7,8],[2,4,6,7]]).to(device)
trg = torch.tensor([[1,3,5],[6,7,3]]).to(device)
src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 10
trg_vocab_size = 10

model = Transformer(src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx).to(device)
out = model(x,trg[:,:-1])
print(out.shape)

