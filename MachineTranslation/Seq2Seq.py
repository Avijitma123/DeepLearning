import random


import torch.nn as nn
import torch

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,v_size,device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.v_size = v_size
        self.device = device

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size =source.shape[1]
        target_len = target.shape[0]

        target_voca_sizr = self.v_size
        outputs = torch.zeros(target_len, batch_size, target_voca_sizr).to(self.device)
        hidden, cell = self.encoder(source)
        # Start token
        x = target[0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        return outputs



