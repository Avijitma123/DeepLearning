import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
import matplotlib.pyplot  as plt
from torchvision.transforms import functional as TF

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

def plot_video_frames(video_frames):
    # Select the first 20 frames
    frames = video_frames[:10]

    # Create a figure with a grid plot
    fig, axs = plt.subplots(2, 5, figsize=(10, 10))

    # Iterate over the frames and plot them
    for i, frame in enumerate(frames):
        # Plot the frame in the corresponding subplot
        ax = axs[i // 5, i % 5]
        ax.imshow(frame.permute(1, 2, 0))
        ax.axis('off')

    # Adjust the spacing and display the plot
    plt.subplots_adjust(hspace=0.3)
    plt.show()