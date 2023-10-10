#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import torch
import numpy as np
from PIL import Image
import pandas as pd
import random 
from rdkit import Chem
from torchvision.transforms import functional

from mol_depict.utils.utils_drawing import draw_molecule_keypoints_rdkit
from mol_depict.utils.utils_image import resize_image
from mol_depict.molfile_parser.label_molfile import LabelMolFile
from molgrapher.utils.utils_augmentation import get_transforms_dict, GraphTransformer
from molgrapher.utils.utils_graph import MolecularGraph
from molgrapher.models.graph_constructor import GraphConstructor


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, config, train=True, predict=False, evaluate=False, hf_dataset=False, *args, **kwargs):
        self.dataset = dataset
        self.config = config
        self.train = train
        self.predict = predict
        self.transforms_dict = get_transforms_dict(config)
        self.evaluate = evaluate
        self.hf_dataset = hf_dataset
        if not self.evaluate:
            self.collate_fn = None
        
    def __len__(self):
        return len(self.dataset)
    
    def decrement_index(self, index):
        if index == 0:
            index += 1
        else:
            index -= 1
        return index

    def get_molecular_graph(self, image, keypoints, molecule):
        # Create graph from ground truth structure
        molecular_graph = MolecularGraph(self.config).from_rdkit_molecule(molecule, keypoints) 
        
        if self.train and (random.random() > 0.1): 
            restricted_proposals = random.random() > 0.1
            # Densify ground truth graph
            graph_constructor = GraphConstructor(
                keypoints, 
                image, 
                config_dataset_keypoint={},
                config_dataset_graph=self.config,
                gt_molecular_graph=molecular_graph, 
                restricted_proposals=restricted_proposals,
                discarded_bond_length_factor=3
            )
            molecular_graph = graph_constructor.augment_gt_molecular_graph()

        if not self.train:
            # Densify ground truth graph
            graph_constructor = GraphConstructor(
                keypoints, 
                image, 
                config_dataset_keypoint={},
                config_dataset_graph=self.config,
                gt_molecular_graph=molecular_graph, 
                restricted_proposals=False,
                discarded_bond_length_factor=1.75
            )
            molecular_graph = graph_constructor.augment_gt_molecular_graph()
        return molecular_graph 

    def __getitem__(self, index, visualization=False):
        while True:
            if self.config["on_fly"]:
                smiles = pd.read_csv(os.path.dirname(__file__) + "/../../data/smiles/experiment-052.csv", header=None, skiprows=index+1, nrows=1)[5].values[0]
                image_filename = os.path.dirname(__file__) + "/../../data/tmp_molecule.png"
                image, keypoints = draw_molecule_keypoints_rdkit(smiles, path=image_filename)
                image = Image.open(image_filename)
                if (image is None) or (keypoints is None):
                    print("Dataset on fly error: image or keypoints not generated")
                    index = self.decrement_index(index)
                    continue
            else:
                # Read keypoints
                keypoints_flat = self.dataset["keypoints"].iloc[index]
                keypoints = [[keypoints_flat[i] - 1, keypoints_flat[i+1] - 1] for i in range(0, len(keypoints_flat), 3)]
                
                # Read image
                if self.hf_dataset:
                    image = Image.open(io.BytesIO(self.dataset["image"].iloc[index]["bytes"])).convert("RGB")
                else:
                    image_filename = self.dataset["image_filename"].iloc[index]
                    image = Image.open(image_filename).convert("RGB") 
                
                if (image.size[0] != self.config["image_size"][1]) or (image.size[1] != self.config["image_size"][2]):
                    # Resize inference images
                    image = resize_image(
                        image, 
                        image_size = (self.config["image_size"][1], self.config["image_size"][2]), 
                        border_size = 30
                    )
                    # Threshold inference images
                    image = np.array(image, dtype=np.float32)/255
                    image = np.stack((image, )*3, axis=-1)
                    image[image > 0.8] = 1.

                else:
                    # Threshold synthetic images
                    image = np.array(image, dtype=np.float32)/255
                    image[image > 0.95] = 1.

                image[image != 1.] = 0.
                
            # Sanity check
            keypoints_x = [x for x, _ in keypoints]
            keypoints_y = [y for _, y in keypoints]
            if (max(keypoints_x) >= self.config["image_size"][1]) or (max(keypoints_y) >= self.config["image_size"][1]) or \
                    (min(keypoints_x) < 0 or (min(keypoints_y)) < 0):
                index = self.decrement_index(index)
                print("Dataset sanity check error: keypoints out of the image")
                continue

            if len(keypoints) < 2:
                index = self.decrement_index(index)
                print("Dataset sanity check error: the image contains only 1 or no keypoints")
                continue
            
            # Keypoints and images augmentations
            if not self.predict: 
                # Both training and validation set are augmented
                # This is an attempt to overcome domain shift.
                transformed = self.transforms_dict["hard"](image=image, keypoints=keypoints)
                if len(keypoints) != len(transformed["keypoints"]):
                    index = self.decrement_index(index)
                    print("Dataset augmentations error: keypoints out of the image")
                    continue
                keypoints = [[int(keypoint[0]), int(keypoint[1])] for keypoint in transformed["keypoints"]]
                image = transformed["image"]
            
            image = torch.from_numpy(image).permute(2, 0, 1)
            
            # Convert molfile to pytorch geometric graph
            if self.hf_dataset:
                molecule = Chem.MolFromMolBlock(self.dataset["mol"].iloc[index], sanitize=False, removeHs=False)
            else:
                molfile_annotator = LabelMolFile(
                    self.dataset["molfile_filename"].iloc[index],
                    reduce_abbreviations = True
                ) 
                molecule = molfile_annotator.rdk_mol
            
            if molecule is None:
                index = self.decrement_index(index)
                print("Dataset error: the Molfile can't be read manually")
                continue

            # Get molecular graph
            molecular_graph = self.get_molecular_graph(image, keypoints, molecule)
            
            if not self.predict:
                # Augment graph
                graph_transformer = GraphTransformer(
                    config = self.config,
                    keypoints_shift_limit = [0, 0.05],
                    decoy_keypoint_shift_limit = [0.05, 0.3],
                    decoy_atom_population_density = 0
                )
                molecular_graph = graph_transformer.augment(molecular_graph)

            molecular_graph.set_nodes_edges_nodes_only()
            data = molecular_graph.to_torch_nodes_only()

            # Invert image for convolutions 0 padding
            image = functional.invert(image)

            if visualization:
                return {
                    "data": data,
                    "images": image,
                    "images_filenames": image_filename,
                    "molecular_graph": molecular_graph,
                }
                            
            if self.evaluate:
                # Scale keypoints
                mask_resolution_factor = 1024/256
                keypoints = [[int(keypoint[0]/mask_resolution_factor), int(keypoint[1]/mask_resolution_factor)] for keypoint in keypoints]

                return {
                    "graph": molecular_graph,
                    "keypoints_batch": keypoints,
                    "images": image,
                    "images_filenames": image_filename,
                    "molfiles_filenames": self.dataset["molfile_filename"].iloc[index]
                }

            return {
                "data": data,
                "images": image
            }
