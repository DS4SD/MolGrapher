#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
import shutil
from rdkit import Chem


def main():
    model_name = "molgrapher_exp-ad03-2"
    benchmark = "uspto_large"
    ylim = [0, 1.05]

    with open(os.path.dirname(__file__) + "/../../../data/scores/molecular_recognition/scores_" + model_name + "_" + benchmark + ".json", 'r') as file:
        scores = json.load(file)

    with open(os.path.dirname(__file__) + "/../../../data/scores/molecular_recognition/information_" + model_name + "_" + benchmark + ".json", 'r') as file:
        molecular_information = json.load(file)   
    
    remove_indices = []
    for image_name in molecular_information.keys():
        molfile_filename = os.path.dirname(__file__) + f"/../../../data/benchmarks/uspto_large/molfiles/{image_name}.MOL"
        molecule = Chem.MolFromMolFile(molfile_filename)
        for atom in molecule.GetAtoms():
            if atom.GetFormalCharge() != 0: 
                remove_indices.append(image_name)
  
    selection = [index for index, molecular_info in molecular_information.items() \
                   if (molecular_info["nb_atoms"] < 1500) and \
                      (molecular_info["stereochemistry"] == False) and \
                      (index not in remove_indices) and \
                      (scores[index]["valid"] == True)
    ]

    scores = {index: score for index, score in scores.items() if index in selection}
    molecular_information = {index: molecular_info for index, molecular_info in molecular_information.items() if index in selection}
    print("Number of test molecule: ", len(scores))

if __name__ == "__main__":
    main()