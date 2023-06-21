import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class new_model(nn.Module):
    def __init__(self, output_layer=None):
        super().__init__()
        self.pretrained = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers) - self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])

        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self, x):
        x = self.net(x)
        return x