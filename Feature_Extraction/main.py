import torch
from Features import Extract


# Define the video path
video_path = "untrimmed_video1.mp4"

feature_extractor = Extract(16)

extracted_feature=feature_extractor.get_feature(video_path)


print("=========Feature vectors============")
print(extracted_feature)


