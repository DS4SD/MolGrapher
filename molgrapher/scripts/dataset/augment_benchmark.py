#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import json 
from tqdm import tqdm
from pprint import pprint
from torchvision.transforms import functional

from torchvision.utils import save_image
from molgrapher.utils.utils_dataset import get_transforms_dict
from molgrapher.data_modules.data_module_graph import GraphDataModule
from molgrapher.datasets.dataset_keypoints import KeypointsDataset


def main():
    dataset = "uspto_large"
    augmented_dataset = "uspto_large_augmented"
    clean = True

    # Read config file
    with open(os.path.dirname(__file__) + "/../../../data/config_dataset_keypoints.json") as file:
        config_dataset_keypoints = json.load(file)

    # Read  dataset
    data_module = GraphDataModule(
        config_dataset_keypoints, 
        dataset_class = KeypointsDataset
    )
    data_module.precompute_benchmarks()
    data_module.setup_benchmarks(train=True)
    
    # Clean 
    if clean:
        if os.path.exists(os.path.dirname(__file__) + f"/../../../data/benchmarks/{augmented_dataset}/images/"): 
            shutil.rmtree(os.path.dirname(__file__) + f"/../../../data/benchmarks/{augmented_dataset}/images/")
        if os.path.exists(os.path.dirname(__file__) + f"/../../../data/benchmarks/{augmented_dataset}/keypoints_images_filenames_{config_dataset_keypoints['experiment_name']}.json"):
            os.remove(os.path.dirname(__file__) + f"/../../../data/benchmarks/{augmented_dataset}/keypoints_images_filenames_{config_dataset_keypoints['experiment_name']}.json")

    # Copy molfiles
    if not os.path.exists(os.path.dirname(__file__) + f"/../../../data/benchmarks/{augmented_dataset}/molfiles/"): 
        shutil.copytree(
            os.path.dirname(__file__) + f"/../../../data/benchmarks/{dataset}/molfiles/",
            os.path.dirname(__file__) + f"/../../../data/benchmarks/{augmented_dataset}/molfiles/"
        )

    # Copy annotations
    if not os.path.exists(os.path.dirname(__file__) + f"/../../../data/benchmarks/{augmented_dataset}/keypoints_images_filenames_{config_dataset_keypoints['experiment_name']}.json"):
        with open(os.path.dirname(__file__) + f"/../../../data/benchmarks/{dataset}/keypoints_images_filenames_{config_dataset_keypoints['experiment_name']}.json", 'r') as json_file:
            coco_json  = json.load(json_file)

        coco_images_filenames = [image["image_filename"] for index, image in enumerate(coco_json["images"]) if index < config_dataset_keypoints["nb_sample_benchmark"]]
        new_coco_images = []
        for image_filename, coco_image in zip(coco_images_filenames, coco_json["images"]):
            coco_image["image_filename"] = image_filename.replace(dataset, augmented_dataset)
            new_coco_images.append(coco_image)
        coco_json["images"] = new_coco_images
        with open(os.path.dirname(__file__) + f"/../../../data/benchmarks/{augmented_dataset}/keypoints_images_filenames_{config_dataset_keypoints['experiment_name']}.json", 'w') as outfile:
            json.dump(coco_json, outfile)

    # Transform images
    loader = data_module.predict_dataloader(dataset)
    images_filenames = [image_filename for batch in loader for image_filename in batch["images_filenames"]]
    images = [image for batch in loader for image in tqdm(batch["images"])]
    
    # Copy images
    if not os.path.exists(os.path.dirname(__file__) + f"/../../../data/benchmarks/{augmented_dataset}/images/"): 
        os.makedirs(os.path.dirname(__file__) + f"/../../../data/benchmarks/{augmented_dataset}/images/")
    for image, image_filename in tqdm(zip(images, images_filenames)):
        image_name = image_filename.split("/")[-1]
        augmented_image_path = os.path.dirname(__file__) + f"/../../../data/benchmarks/{augmented_dataset}/images/{image_name}"

        # Invert back image
        image = functional.invert(image)
        save_image(image, augmented_image_path)
    

            


if __name__ == "__main__":
    main()