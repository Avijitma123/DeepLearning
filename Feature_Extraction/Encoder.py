import torch
import torch.nn as nn


# Define the feature projection layers
class FeatureProjectionLayers(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureProjectionLayers, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


# Define the Temporal Context Modeling (TCM) block
class TCMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TCMBlock, self).__init__()

        # Define your TCM block layers here (e.g., convolutional layers, pooling, etc.)
        # You can customize the structure based on your requirements

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)

        return out


# Define the encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_tcm_blocks):
        super(Encoder, self).__init__()

        self.feature_projection = FeatureProjectionLayers(in_channels, out_channels)
        self.tcm_blocks = nn.ModuleList([TCMBlock(out_channels, out_channels) for _ in range(num_tcm_blocks)])

    def forward(self, x):
        # Feature projection layers
        out = self.feature_projection(x)

        # Temporal Context Modeling (TCM) blocks
        for tcm_block in self.tcm_blocks:
            out = tcm_block(out)

        return out