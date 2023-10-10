#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional

from mol_depict.utils.utils_image import resize_image
from molgrapher.utils.utils_dataset import crop_tight


class MolfileDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        dataset, 
        config, 
        train=False, 
        predict=True, 
        return_images_filenames=False, 
        preprocessed=False, 
        evaluate=False, 
        force_cpu=False, 
        taa_step=0,
        *args,
        **kwargs
    ):
        self.dataset = dataset
        self.config = config
        self.return_images_filenames = return_images_filenames
        self.border_size = 30
        self.preprocessed = preprocessed
        self.evaluate = evaluate
        self.taa_step = taa_step
        if predict:
            from molgrapher.utils.utils_dataset import CaptionRemover
            self.caption_remover = CaptionRemover(force_cpu = force_cpu)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        if self.return_images_filenames:
            return {
                "images_filenames": self.dataset["image_filename"].iloc[index],
                "molfiles_filenames": self.dataset["molfile_filename"].iloc[index]
            }
        else:
            image_filename = self.dataset["image_filename"].iloc[index]
            pil_image = Image.open(image_filename)
            
            if not self.preprocessed:
                pil_image = resize_image(
                    pil_image, 
                    (self.config["image_size"][1], self.config["image_size"][1]), 
                    border_size = self.border_size
                )
            
                # Remove captions (95% of the dataloading time)
                image = self.caption_remover(pil_image)
                
                # Remove borders
                pil_image = Image.fromarray(image).convert('RGB')
                pil_image = crop_tight(pil_image)
                
                if self.taa_step == 0:
                    # Resize, add small borders 
                    pil_image = resize_image(
                        pil_image,
                        (self.config["image_size"][1], self.config["image_size"][1]), 
                        border_size = self.border_size*3 
                    )
                if self.taa_step == 1:
                    # Resize, add small borders 
                    pil_image = resize_image(
                        pil_image,
                        (self.config["image_size"][1], self.config["image_size"][1]), 
                        border_size = self.border_size*8 
                    )

                if self.taa_step == 2:
                    # Resize, add small borders 
                    pil_image = resize_image(
                        pil_image,
                        (self.config["image_size"][1], self.config["image_size"][1]), 
                        border_size = self.border_size*5
                    )
               
            # Threshold and convert to float
            image = np.array(pil_image, dtype=np.float32)/255
            image[image > 0.6] = 1.
            image[image != 1.] = 0.
            image = np.stack((image, )*3, axis=-1)

            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1)
            
            # Invert image for convolutions 0 padding
            image = functional.invert(image)
            
            return {
                "images": image,
                "images_filenames": self.dataset["image_filename"].iloc[index],
                "molfiles_filenames": self.dataset["molfile_filename"].iloc[index]
            }
        
