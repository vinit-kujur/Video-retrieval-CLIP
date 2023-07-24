"""
Author: Vinit Kujur
"""


import torch
import clip

from torch import nn
import numpy as np
import pickle
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model and image preprocessing
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)


def calculate_similarity(image_path, video_features):

    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        query_features = model.encode_image(image)

    # Load data (deserialize)
    with open(video_features, 'rb') as handle:
        all_image_features = pickle.load(handle)


    similarity = {}
    # iterate over image features in that directory
    for file_name in all_image_features:
        image_features = all_image_features[file_name]
        with torch.no_grad():
            logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            query_features = query_features / query_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            # logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features[0] @ query_features.t()  # can scale it!

            # sim_scores.append(logits_per_image.item())
            similarity[file_name] = logits_per_image.item()

    similarity_sorted = {k: v for k, v in sorted(similarity.items(), key=lambda item: item[1])}
    for file in similarity_sorted:
        frame_no = file.split('.')[0]
        frame_no = int(frame_no)
        time_stamp = frame_no//25
        minutes = time_stamp // 60
        seconds = time_stamp % 60
        print(f"{file}, {similarity_sorted[file]}, {minutes}:{seconds}")

    return similarity


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print("In Main...")
    file_name = "video_meta/wildlife2"
    out_file = file_name + ".pickle"
    out_file = "video_meta/" + ".pickle"
    

    while True:
        image_path = input("Image Path: ")
        similarity = calculate_similarity(image_path, out_file)

