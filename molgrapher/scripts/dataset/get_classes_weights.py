#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json 
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import cv2 
import numpy as np

from molgrapher.data_modules.data_module import DataModule
from molgrapher.datasets.dataset_graph import GraphDataset

os.environ["OMP_NUM_THREADS"] = "4" 
cv2.setNumThreads(0)

def main():

    # Read config files
    with open(os.path.dirname(__file__) + "/../../../data/config_dataset_graph_2.json") as file:
        config_dataset = json.load(file)

    with open(os.path.dirname(__file__) + "/../../../data/config_training_graph_2.json") as file:
        config_training = json.load(file)

    # Read  dataset
    data_module = DataModule(config_dataset, dataset_class = GraphDataset)
    #data_module.precompute_keypoints_synthetic()
    data_module.setup()

    loader = data_module.train_dataloader()

    symbols_classes_atoms = json.load(open(os.path.dirname(__file__) + "/../../../data/vocabularies/vocabulary_atoms.json"))
    types_classes_bonds = json.load(open(os.path.dirname(__file__) + "/../../../data/vocabularies/vocabulary_bonds.json"))
    atoms_classes_symbols = {v: k for k,v in symbols_classes_atoms.items()}
    bonds_classes_types = {v: k for k,v in types_classes_bonds.items()}

    atoms_classes_population = {v: 1 for k,v in symbols_classes_atoms.items()}
    bonds_classes_population = {v: 1 for k,v in types_classes_bonds.items()}

    max_index = 10000
    for index in tqdm(range(min(len(loader)*config_dataset["batch_size"], max_index))):
        sample = loader.dataset.__getitem__(index, visualization=True)
        data, image, molecular_graph, image_filename = sample["data"], sample["images"], sample["molecular_graph"], sample["images_filenames"]
        
        for atom in molecular_graph.atoms:
            atoms_classes_population[atom["class"]] += 1

        for bond in molecular_graph.bonds:
            bonds_classes_population[bond["class"]] += 1
    
    atoms_symbols_population = {atoms_classes_symbols[k]: v for k, v in atoms_classes_population.items()}
    bonds_symbols_population = {bonds_classes_types[k]: v for k, v in bonds_classes_population.items()}

    atoms_threshold = np.percentile(list(atoms_symbols_population.values()), 20)
    bonds_threshold = np.percentile(list(bonds_symbols_population.values()), 20)

    atoms_weights = {k: 1 if v > atoms_threshold else 3 for k, v in atoms_symbols_population.items()}
    bonds_weights = {k: 1 if v > bonds_threshold else 3 for k, v in bonds_symbols_population.items()}

    with open(f"{config_dataset['experiment_name']}_atoms_weights.json", "w") as outfile:
        json.dump(atoms_weights, outfile)
        
    with open(f"{config_dataset['experiment_name']}_bonds_weights.json", "w") as outfile:
        json.dump(bonds_weights, outfile)

    print("Populations:")
    print(atoms_symbols_population)
    print(bonds_symbols_population)

    print("Weights:")
    print(atoms_weights)
    print(bonds_weights)

if __name__ == "__main__":
    main()