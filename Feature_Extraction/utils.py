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
    frames = video_frames[:20]

    # Create a figure with a grid plot
    fig, axs = plt.subplots(4, 5, figsize=(10, 10))

    # Iterate over the frames and plot them
    for i, frame in enumerate(frames):
        # Plot the frame in the corresponding subplot
        ax = axs[i // 5, i % 5]
        ax.imshow(frame.permute(1, 2, 0))
        ax.axis('off')
        ax.set_title(f"Frame {i+1}")

    # Adjust the spacing and display the plot
    plt.subplots_adjust(hspace=0.3)
    plt.show()
def plot_output_featur_map(features):
    # Plot feature maps using a grid layout
    # feature dimention is [ 512, 21, 7 , 7]

    # Extract the feature maps from the model
    feature_maps = features.squeeze().numpy()  # Shape: (512, 21, 7, 7)

    # Define the number of feature maps and filters per map
    num_feature_maps = feature_maps.shape[1]-260
    num_filters_per_map = min(feature_maps.shape[0], 5)  # Limit to 10 filters per feature map

    # Plot feature maps using a grid layout
    grid_size = (num_filters_per_map, num_feature_maps)
    fig, axes = plt.subplots(*grid_size, figsize=(10, 10))

    for i in range(num_feature_maps):
        for j in range(num_filters_per_map):
            axes[j, i].imshow(feature_maps[j, i], cmap='gray')
            axes[j, i].axis('off')
            axes[j, i].set_title(f'F {j + 1}')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def VisualizeR3D_Model():
    children_counter = 0
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    for n,c in model.named_children():
       print("Children Counter: ",children_counter," Layer Name: ",n,)
       children_counter+=1
