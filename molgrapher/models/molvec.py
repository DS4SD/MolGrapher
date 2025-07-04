#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
import os
import subprocess

import pytorch_lightning as pl
from rdkit import Chem


def predict_process(image_filename):
    jar_path = os.path.dirname(__file__) + "/../../data/external/jar/"
    class_path = (
        jar_path
        + "molvec-0.9.8.jar:"
        + jar_path
        + "common-image-3.6.jar:"
        + jar_path
        + "common-io-3.6.jar:"
        + jar_path
        + "common-lang-3.6.jar:"
        + jar_path
        + "commons-cli-1.4.jar:"
        + jar_path
        + "imageio-core-3.6.jar:"
        + jar_path
        + "imageio-metadata-3.6.jar:"
        + jar_path
        + "imageio-tiff-3.6.jar:"
        + jar_path
        + "ncats-common-0.3.4.jar:"
        + jar_path
        + "ncats-common-cli-0.9.2.jar"
    )

    molfile_path = (
        os.path.dirname(__file__)
        + "/../../data/predictions/molvec/"
        + image_filename.split("/")[-1][:-4]
        + ".MOL"
    )
    process = subprocess.Popen(
        "exec java -cp "
        + class_path
        + " gov.nih.ncats.molvec.Main -f "
        + image_filename
        + " > "
        + molfile_path,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        outs, errors = process.communicate(timeout=50)
    except subprocess.TimeoutExpired:
        process.kill()
        return None
    if errors != b"":
        print(errors)
        return None

    molecule = Chem.MolFromMolFile(molfile_path)
    if molecule is None:
        return None

    smiles = Chem.MolToSmiles(molecule)
    if smiles is None:
        return None
    return smiles


class MolVec(pl.LightningModule):
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
