#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pytorch_lightning as pl
from img2mol.inference import *


class Img2Mol(pl.LightningModule):
    def __init__(self):
        super().__init__()
        img2mol_path = os.path.join(os.path.dirname(__file__), "../../../my-img2mol/")
        self.img2mol_model = Img2MolInference(
            model_ckpt=img2mol_path + "model/model.ckpt", device="cpu"
        )
        self.cddd_server = CDDDRequest()

    def predict(self, images_filenames):
        # Load Img2Mol model

        smiles_batch = []
        for image_path in images_filenames:
            smiles_batch.append(
                self.img2mol_model(filepath=image_path, cddd_server=self.cddd_server)[
                    "smiles"
                ]
            )

        return smiles_batch

    def predict_step(self, batch, batch_idx):
        return self.predict(batch["images_filenames"])
