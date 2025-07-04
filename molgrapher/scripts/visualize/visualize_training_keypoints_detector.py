#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil

import matplotlib.pyplot as plt
from pytorch_lightning.trainer.states import TrainerFn
from torchvision import transforms
from tqdm import tqdm

from molgrapher.data_modules.data_module import DataModule
from molgrapher.datasets.dataset_keypoint import KeypointDataset
from molgrapher.datasets.dataset_keypoint_cede import KeypointCedeDataset


def main():
    fine_tunning = False
    precompute = False
    clean = True
    output_folder = "test1"

    if clean:
        if os.path.exists(
            os.path.dirname(__file__)
            + f"/../../../data/visualization/training_images/keypoint_detector/{output_folder}"
        ):
            shutil.rmtree(
                os.path.dirname(__file__)
                + f"/../../../data/visualization/training_images/keypoint_detector/{output_folder}/"
            )
    if not os.path.exists(
        os.path.dirname(__file__)
        + f"/../../../data/visualization/training_images/keypoint_detector/{output_folder}"
    ):
        os.makedirs(
            os.path.dirname(__file__)
            + f"/../../../data/visualization/training_images/keypoint_detector/{output_folder}"
        )

    # Read config file
    with open(
        os.path.dirname(__file__) + "/../../../data/config_dataset_keypoint.json"
    ) as file:
        config_dataset = json.load(file)

    # Read config file
    with open(
        os.path.dirname(__file__) + "/../../../data/config_training_keypoint.json"
    ) as file:
        config_training = json.load(file)

    # Read  dataset
    if fine_tunning:
        mode = "fine-tunning"
    else:
        mode = "train"
    if config_dataset["training_dataset"] == "cede-synthetic":
        print("Create Datamodule for CEDe keypoints")
        data_module = DataModule(
            config=config_dataset,
            dataset_class=KeypointCedeDataset,
            mode=mode,
            force_precompute=False,
        )
    else:
        print("Create datamodule for MolGrapher keypoints")
        data_module = DataModule(
            config=config_dataset,
            dataset_class=KeypointDataset,
            mode=mode,
            force_precompute=False,
        )

    if precompute:
        data_module.precompute_keypoints_synthetic()
    data_module.setup(stage=TrainerFn.FITTING)

    loader = data_module.train_dataloader()

    scaling_factor = config_dataset["image_size"][1] / config_dataset["mask_size"][1]
    t = transforms.Compose([transforms.Resize([1024, 1024])])

    max_index = 100
    for index in tqdm(
        range(min(len(loader) * config_dataset["batch_size"], max_index))
    ):
        sample = loader.dataset.__getitem__(index)
        image, keypoints, image_filename, heatmap = (
            sample["images"],
            sample["keypoints_batch"],
            sample["images_filenames"],
            sample["heatmaps_batch"],
        )
        image = image.permute(1, 2, 0).numpy()[:, :, 0]
        # heatmap = torch.nn.Threshold(0.99999, 255)(heatmap)
        figure, axis = plt.subplots(1, 2, figsize=(14, 6))
        axis[0].imshow(image, cmap="gray")
        axis[0].scatter(
            [
                (keypoint[0] * scaling_factor + scaling_factor // 2)
                for keypoint in keypoints
            ],
            [
                (keypoint[1] * scaling_factor + scaling_factor // 2)
                for keypoint in keypoints
            ],
            color="red",
        )
        im2 = axis[1].imshow(t(heatmap).permute(1, 2, 0))

        axis[1].imshow(1 - image, alpha=0.25 * (image < 1), cmap="gray")

        plt.savefig(
            os.path.dirname(__file__)
            + f"/../../../data/visualization/training_images/keypoint_detector/{output_folder}/{image_filename.split('/')[-1]}"
        )
        plt.close()


if __name__ == "__main__":
    main()
