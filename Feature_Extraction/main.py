import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.transforms import functional as TF
from torchvision.io.video import read_video
from torchsummary import summary
from utils import new_model,plot_video_frames

i3d_model = new_model(output_layer='layer4')

# Set the model to evaluation mode
i3d_model.eval()

# Define the video path
video_path = "video1.avi"

# Read the video frames
video_frames, audio, info = read_video(video_path, pts_unit="sec")  # Returns frames in shape (T, H, W, C)

# Convert video frames to tensor and normalize
video_tensor = video_frames.permute(0, 3, 1, 2).float()  # Shape: (T, C, H, W)

# Preprocess the input video frames
resized_frames = []
for frame in video_tensor:
    resized_frame = TF.resize(frame, [128, 171], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    cropped_frame = TF.center_crop(resized_frame, [112, 112])
    resized_frames.append(cropped_frame)

preprocessed_video = torch.stack(resized_frames)  # Shape: (T, C, H, W)

print("Preprocessed video shape:",preprocessed_video.shape)
#plot_video_frames(preprocessed_video)
# Rescale and normalize the preprocessed video frames
rescaled_video = preprocessed_video / 255.0

# Verify the shape of rescaled_video
print("Rescaled video shape:", rescaled_video.shape)

# Calculate the mean and standard deviation per channel
mean = rescaled_video.mean(dim=[0, 2, 3])
std = rescaled_video.std(dim=[0, 2, 3])

# Normalize the video frames
normalized_video = TF.normalize(rescaled_video, mean=mean, std=std)

# Permute the dimensions of the video tensor
input_tensor = normalized_video.permute(1, 0, 2, 3).unsqueeze(0)  # Shape: (1, T, C, H, W)
print("Input Tensor:", input_tensor.shape)

# Pass the video frames through the I3D model
with torch.no_grad():
    features = i3d_model(input_tensor)  # Extract features from the stem module



# summary(i3d_model, input_size=(3, 164, 112, 112))
print("Feature vector shape:", features.shape)
print("Feature vector:", features)




