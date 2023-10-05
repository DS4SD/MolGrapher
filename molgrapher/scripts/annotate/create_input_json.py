#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import glob 
import os 

dataset = "default"
dataset_size = 10000
image_folder_path = os.path.dirname(__file__) + f"/../../../data/benchmarks/{dataset}/images/"
output_file_name = f"input_images_paths_{dataset}.jsonl"
images_filenames = []

for image_filename in glob.glob(image_folder_path + "*"):
    if ("preprocessed" not in image_filename) and (".png" in image_filename):
        images_filenames.append(
            {"path": image_filename}
        )
images_filenames = images_filenames[:dataset_size]

with open(output_file_name, "w") as f:
    for image_filename in images_filenames:
        json.dump(image_filename, f)
        f.write("\n")