#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch
from huggingface_hub import hf_hub_download
from molscribe import MolScribe
from tqdm import tqdm


class Molscribe:
    def __init__(self, device):
        super().__init__()
        ckpt_path = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
        self.model = MolScribe(ckpt_path, device=device)

    def predict(self, images_filenames):
        smiles_batch = []
        for image_path in tqdm(images_filenames):
            print(image_path)
            output = self.model.predict_image_file(image_path)
            smiles_batch.append(output["smiles"])
            print(output["smiles"])
        return smiles_batch
