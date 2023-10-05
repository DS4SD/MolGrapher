#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import pytorch_lightning as pl
import multiprocessing
from rdkit import Chem 

def predict_process(image_filename):
    process = subprocess.Popen("exec osra -i " + image_filename, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        outs, errors = process.communicate(timeout=50)
    except subprocess.TimeoutExpired:
        process.kill()
        return None

    if (errors != b""):
        print(errors)
        return None
    smiles = outs.decode('UTF-8')
    smiles = smiles.replace("\n", "")
    return smiles

class OSRA(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def predict(self, images_filenames):
        nb_processes = len(images_filenames)
        pool = multiprocessing.Pool(nb_processes)
        smiles_batch = pool.map(predict_process, images_filenames)
        pool.close()
        pool.join()
        return smiles_batch
    
    def predict_step(self, batch, batch_idx):
        return self.predict(batch["images_filenames"])
