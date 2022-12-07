"""
Author: Vinit Kujur
"""

import os
import torch
import clip
from PIL import Image

import pickle
import argparse

device = "cuda:2" if torch.cuda.is_available() else "cpu"

# load model and image preprocessing
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

out_folder = "extracted/"


def get_image_features():
    all_image_features = {}
    # iterate over files in that directory
    for file_name in os.listdir(out_folder):
        file_path = os.path.join(out_folder, file_name)
        # checking if it is a file
        if os.path.isfile(file_path):
            # do the processing...
            print(f"Processing: {file_path}")
            # preprocess the image
            image = Image.open(file_path)
            image = preprocess(image)
            image = image.unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            all_image_features[file_name] = image_features

    return all_image_features


def extract_keyframes(video_file):
    """
    :param video_file: path of the video file
    :return:
    """
    out_file = f"%d.png"
    out_file_path = out_folder + out_file
    command = f"ffmpeg -skip_frame nokey -i {video_file} -vsync 0 -frame_pts true {out_file_path}"
    os.system(command)

    pass


def save_image_features(video_file, file_name):
    extract_keyframes(video_file)
    all_image_features = get_image_features()
    folder = "video_meta/"
    with open(folder + file_name, 'wb') as handle:
        pickle.dump(all_image_features, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Argparse declaration

    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True, help="video file")
    args = vars(ap.parse_args())

    video_file = args["file"]
    extract_keyframes(video_file)
    print("Processing...")

    file_name = video_file.split('/')[-1]  # without extension
    file_name = file_name.split('.')[0]  # without extension
    file_name = file_name + ".pickle"
    save_image_features(video_file, file_name)


