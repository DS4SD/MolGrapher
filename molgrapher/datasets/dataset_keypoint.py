#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os

import numpy as np
import pandas as pd
import torch
from mol_depict.utils.utils_drawing import draw_molecule_keypoints_rdkit
from mol_depict.utils.utils_image import resize_image
from PIL import Image
from torchvision.transforms import functional

from molgrapher.utils.utils_augmentation import (GraphTransformer,
                                                 get_transforms_dict)
from molgrapher.utils.utils_dataset import get_bond_size


class KeypointDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        config,
        train=True,
        predict=False,
        evaluate=False,
        hf_dataset=False,
        *args,
        **kwargs
    ):
        self.dataset = dataset
        self.config = config
        self.train = train
        self.predict = predict
        self.transforms_dict = get_transforms_dict(config)
        self.evaluate = evaluate
        self.hf_dataset = hf_dataset
        self.precomputed_gaussians = {}

    def decrement_index(self, index):
        if index == 0:
            index += 1
        else:
            index -= 1
        return index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        while True:
            if self.config["on_fly"]:
                drawing_successful = False
                while not drawing_successful:
                    smiles = pd.read_csv(
                        os.path.dirname(__file__)
                        + "/../../data/smiles/experiment-052.csv",
                        header=None,
                        skiprows=index + 1,
                        nrows=1,
                    )[5].values[0]
                    # print("SMILES: ", smiles) # Debug
                    image_filename = (
                        os.path.dirname(__file__) + "/../../data/tmp_molecule.png"
                    )
                    image, keypoints = draw_molecule_keypoints_rdkit(
                        smiles, path=image_filename
                    )
                    image = Image.open(image_filename)
                    if (image is not None) and (keypoints is not None):
                        index = self.decrement_index(index)
                        continue
            else:
                # Read keypoints
                keypoints_flat = self.dataset["keypoints"].iloc[index]
                keypoints = [
                    (keypoints_flat[i], keypoints_flat[i + 1])
                    for i in range(0, len(keypoints_flat), 3)
                ]

                # Read image
                if self.hf_dataset:
                    image = Image.open(
                        io.BytesIO(self.dataset["image"].iloc[index]["bytes"])
                    ).convert("RGB")
                else:
                    image_filename = self.dataset["image_filename"].iloc[index]
                    image = Image.open(image_filename).convert("RGB")

                if (image.size[0] != self.config["image_size"][1]) or (
                    image.size[1] != self.config["image_size"][2]
                ):
                    # Resize inference image
                    image = resize_image(
                        image,
                        image_size=(
                            self.config["image_size"][1],
                            self.config["image_size"][2],
                        ),
                        border_size=30,
                    )
                    # Threshold
                    image = np.array(image, dtype=np.float32) / 255

                    image[image > 0.8] = 1.0
                    image = np.stack((image,) * 3, axis=-1)
                else:
                    # Threshold synthetic images
                    image = np.array(image, dtype=np.float32) / 255
                    image[image > 0.95] = 1.0

                image[image != 1.0] = 0.0

            # Sanity check
            keypoints_x = [x for x, _ in keypoints]
            keypoints_y = [y for _, y in keypoints]
            if (
                (max(keypoints_x) >= self.config["image_size"][1])
                or (max(keypoints_y) >= self.config["image_size"][1])
                or (min(keypoints_x) < 0 or (min(keypoints_y)) < 0)
            ):
                index = self.decrement_index(index)
                print("Dataset sanity check error: keypoints out of the image")
                continue

            # Augmentations
            if not self.predict:
                # Both training and validation set are augmented
                # This is an attempt to overcome domain shift.
                transforms = self.transforms_dict["extreme"]
                transformed = transforms(image=image, keypoints=keypoints)
                if len(keypoints) != len(transformed["keypoints"]):
                    index = self.decrement_index(index)
                    print("Dataset augmentations error: keypoints out of the image")
                    continue
                keypoints = [
                    [int(keypoint[0]), int(keypoint[1])]
                    for keypoint in transformed["keypoints"]
                ]
                image = transformed["image"]

                # Blurring can results in values greater than 1
                image[np.where(image > 1)] = 1

            # Estimate bond size
            self.bond_size = get_bond_size(keypoints)

            # Augment keypoints positions
            if self.train:
                graph_transformer = GraphTransformer(
                    config=self.config,
                    keypoints_shift_limit=[0, 0.01],
                    decoy_keypoint_shift_limit=[0.05, 0.3],
                    decoy_atom_population_density=0,
                    keypoints_only=True,
                )

                keypoints = graph_transformer.shift_keypoints_positions(
                    keypoints,
                    shift_window=(
                        graph_transformer.keypoints_shift_limit[0] * self.bond_size,
                        graph_transformer.keypoints_shift_limit[1] * self.bond_size,
                    ),
                )

            image = torch.from_numpy(image).permute(2, 0, 1)

            # Scale keypoints
            mask_resolution_factor = (
                self.config["image_size"][1] / self.config["mask_size"][1]
            )
            keypoints = [
                [
                    int(keypoint[0] / mask_resolution_factor),
                    int(keypoint[1] / mask_resolution_factor),
                ]
                for keypoint in keypoints
            ]

            # Convert keypoints to heatmap
            heatmap = self.generate_heatmap(keypoints)
            if heatmap is None:
                index = self.decrement_index(index)

            heatmap = torch.Tensor(heatmap).unsqueeze(0)

            # Invert image for convolutions 0 padding
            image = functional.invert(image)
            return {
                "images_filenames": self.dataset["image_filename"].iloc[index],
                "images": image,
                "keypoints_batch": keypoints,
                "heatmaps_batch": heatmap,
            }

    def add_gaussian(self, input, keypoint, sigma):
        tmp_size = sigma * 3

        # Top-left
        x1, y1 = int(keypoint[0] - tmp_size), int(keypoint[1] - tmp_size)

        # Bottom right
        x2, y2 = int(keypoint[0] + tmp_size + 1), int(keypoint[1] + tmp_size + 1)
        if x1 >= input.shape[0] or y1 >= input.shape[1] or x2 < 0 or y2 < 0:
            return None

        size = 2 * tmp_size + 1
        tx = np.arange(0, size, 1, np.float32)
        ty = tx[:, np.newaxis]
        x0 = y0 = size // 2

        g = (
            self.precomputed_gaussians[sigma]
            if sigma in self.precomputed_gaussians
            else torch.tensor(
                np.exp(-((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma**2))
            )
        )
        self.precomputed_gaussians[sigma] = g

        # Determine the bounds of the source gaussian
        g_x_min, g_x_max = max(0, -x1), min(x2, input.shape[0]) - x1
        g_y_min, g_y_max = max(0, -y1), min(y2, input.shape[1]) - y1

        # Image range
        img_x_min, img_x_max = max(0, x1), min(x2, input.shape[0])
        img_y_min, img_y_max = max(0, y1), min(y2, input.shape[1])

        input[img_y_min:img_y_max, img_x_min:img_x_max] = torch.max(
            input[img_y_min:img_y_max, img_x_min:img_x_max],
            g[g_y_min:g_y_max, g_x_min:g_x_max],
        )
        return input

    def generate_heatmap(self, keypoints):
        sigma = max(round(self.bond_size / 40), 2)

        heatmap = torch.zeros(self.config["mask_size"][1], self.config["mask_size"][2])
        for keypoint in keypoints:
            heatmap = self.add_gaussian(heatmap, keypoint, sigma)
            if heatmap is None:
                return None
        return heatmap

    def collate_fn(self, batch):
        new = {k: [] for k in batch[0].keys()}

        for item in batch:
            for key, value in item.items():
                new[key].append(value)

        new["images"] = torch.stack(new["images"])
        new["heatmaps_batch"] = torch.stack(new["heatmaps_batch"])
        return new
