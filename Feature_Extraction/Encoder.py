# Define the encoder architecture
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, num_levels):
        super(Encoder, self).__init__()
        self.num_levels = num_levels

        # Feature projection layers
        self.E1 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=1)
        self.E2 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=1)

        # Temporal Context Modeling (TCM) blocks
        self.TCM_blocks = nn.ModuleList([self._create_TCM_block() for _ in range(num_levels - 1)])

    def _create_TCM_block(self):
        return nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, X):
        # Feature projection layers
        X_e1 = self.E1(X)
        X_e2 = self.E2(X_e1)

        # Concatenate X with E1 output along the channel dimension
        X_concat = torch.cat((X, X_e1), dim=1)

        # Intermediate output of feature projection layers
        Xp = X_concat

        # Temporal Context Modeling (TCM) blocks
        Z = [Xp]  # Feature pyramid

        for i in range(self.num_levels - 1):
            X_tcm = self.TCM_blocks[i](Xp)
            Xp = torch.cat((Xp, X_tcm), dim=1)
            Z.append(Xp)

        return Z
