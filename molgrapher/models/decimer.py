#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import DECIMER 
import pytorch_lightning as pl


class Decimer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def predict(self, images_filenames):
        smiles_batch = []

        for image_path in images_filenames:
            smiles_batch.append(DECIMER.predict_SMILES(image_path))
        print(smiles_batch)
        return smiles_batch
    
    def predict_step(self, batch, batch_idx):
        return self.predict(batch["images_filenames"])
    