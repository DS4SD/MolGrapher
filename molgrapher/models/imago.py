#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import pytorch_lightning as pl
import multiprocessing
import os
from rdkit import Chem


def predict_process(image_filename):
    molfile_path = os.path.dirname(__file__) + "/../../data/predictions/imago/" + image_filename.split("/")[-1][:-4] + ".MOL"
    process = subprocess.Popen("exec " + os.path.dirname(__file__) + "/../../data/external/bin/imago_console " + image_filename + " -o " + molfile_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        outs, errors = process.communicate(timeout=50)
    except subprocess.TimeoutExpired:
        process.kill()
        return None
    if (errors != b""):
        print(errors)
        return None
    
    molecule = Chem.MolFromMolFile(molfile_path)
    
    if molecule is None:
        return None
    
    Chem.RemoveStereochemistry(molecule)
    smiles = Chem.MolToSmiles(molecule)
    
    if smiles is None:
        return None
    
    return smiles

class Imago(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def predict(self, images_filenames):
        predict_process(images_filenames[0])
        nb_processes = len(images_filenames)
        pool = multiprocessing.Pool(nb_processes)
        smiles_batch = pool.map(predict_process, images_filenames)
        pool.close()
        pool.join()
        return smiles_batch
    
    def predict_step(self, batch, batch_idx):
        return self.predict(batch["images_filenames"])
