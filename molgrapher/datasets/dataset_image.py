#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import time

import numpy as np
import torch
from mol_depict.utils.utils_image import resize_image
from PIL import Image
from torchvision.transforms import functional

from molgrapher.utils.utils_dataset import CaptionRemover, crop_tight


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        config,
        preprocessed=False,
        evaluate=False,
        force_cpu=False,
        *args,
        **kwargs
    ):
        self.dataset = dataset
        self.config = config
        self.border_size = 30
        self.evaluate = evaluate
        self.caption_remover = CaptionRemover(force_cpu=force_cpu)
        self.preprocessed = preprocessed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        time_s = time()
        image_filename = self.dataset["image_filename"].iloc[index]
        pil_image = self.dataset["image"].iloc[index]
        if pil_image is None:
            pil_image = Image.open(image_filename)

        if not self.preprocessed:
            pil_image = resize_image(
                pil_image,
                (self.config["image_size"][1], self.config["image_size"][1]),
                border_size=self.border_size,
                b="white",
            )

            # Remove captions
            image = self.caption_remover(pil_image)

            # Remove borders
            pil_image = Image.fromarray(image).convert("RGB")
            pil_image = crop_tight(pil_image)

            # Resize, add small borders
            pil_image = resize_image(
                pil_image,
                (self.config["image_size"][1], self.config["image_size"][1]),
                border_size=self.border_size * 3,
            )

        # Threshold and convert to float
        image = np.array(pil_image, dtype=np.float32) / 255
        image[image > 0.6] = 1.0
        image[image != 1.0] = 0.0
        image = np.stack((image,) * 3, axis=-1)

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Invert image for convolutions 0 padding
        image = functional.invert(image)

        return {"images": image, "images_filenames": image_filename}
