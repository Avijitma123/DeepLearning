import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.transforms import functional as TF
from torchvision.io.video import read_video

# Load the pretrained I3D model
i3d_model = r3d_18(weights=R3D_18_Weights.DEFAULT)

# Set the model to evaluation mode
i3d_model.eval()

"""
The video is divided into k-frame clips using the torch.split() function, 
and then each clip is passed through the I3D model to extract features. 
The extracted features for each clip are stored in a list, and at the end, 
all the features are stacked into a single tensor using torch.stack(). 
The shape of the features tensor is printed along with the feature values.
"""
class Extract:
    def __init__(self,number_clips):
        self.number_clips = number_clips


    def get_feature(self,video_path):

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

        # Rescale and normalize the preprocessed video frames
        rescaled_video = preprocessed_video / 255.0

        # Verify the shape of rescaled_video
        print("Rescaled video shape:", rescaled_video.shape)

        # Calculate the mean and standard deviation per channel
        mean = rescaled_video.mean(dim=[0, 2, 3])
        std = rescaled_video.std(dim=[0, 2, 3])

        # Normalize the video frames
        normalized_video = TF.normalize(rescaled_video, mean=mean, std=std)

        # Divide the video into 16-frame clips
        clip_length = self.number_clips
        num_frames = normalized_video.shape[0]
        num_clips = num_frames // clip_length

        clips = torch.split(normalized_video[:num_clips * clip_length], clip_length)

        # Extract features for each clip
        features = []
        for clip in clips:
            # Permute the dimensions of the video tensor
            input_tensor = clip.permute(1, 0, 2, 3).unsqueeze(0)  # Shape: (1, T, C, H, W)

            # Pass the video frames through the I3D model
            with torch.no_grad():
                clip_features = i3d_model.stem(input_tensor)  # Extract features from the stem module

            # Apply adaptive average pooling to obtain the feature vector
            adaptive_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            clip_feature_vector = adaptive_avg_pool(clip_features).squeeze()  # Shape: (C,)

            features.append(clip_feature_vector)

        # Stack the extracted features
        features = torch.stack(features)

        return features



