#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from PIL import Image
import random 
from torchvision.transforms import functional
import cv2

from molgrapher.utils.utils_augmentation import get_transforms_dict, GraphTransformer
from molgrapher.utils.utils_graph import MolecularGraph
from molgrapher.models.graph_constructor import GraphConstructor


class GraphCedeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, config, train=True, predict=False, evaluate=False, *args, **kwargs):
        self.dataset = dataset
        self.config = config
        self.train = train
        self.predict = predict
        self.evaluate = evaluate
        self.transforms_dict = get_transforms_dict(config)

    def __len__(self):
        return len(self.dataset)
    
    def decrement_index(self, index):
        if index == 0:
            index += 1
        else:
            index -= 1
        return index

    def get_molecular_graph(self, image, keypoints, annotation):
        # Create graph from ground truth structure
        molecular_graph = MolecularGraph(self.config).from_cede(annotation, keypoints) # 15% max potential speed up 
        
        if self.train and (random.random() > 0.1): 
            restricted_proposals = random.random() > 0.1
            # Densify ground truth graph
            graph_constructor = GraphConstructor(
                keypoints, 
                image, 
                self.config, 
                gt_molecular_graph = molecular_graph, 
                restricted_proposals = restricted_proposals,
                discarded_bond_length_factor = 3
            )
            molecular_graph = graph_constructor.augment_gt_molecular_graph()

        if not self.train:
            # Densify ground truth graph
            graph_constructor = GraphConstructor(
                keypoints, 
                image, 
                self.config, 
                gt_molecular_graph = molecular_graph, 
                restricted_proposals = False,
                discarded_bond_length_factor = 1.75
            )
            molecular_graph = graph_constructor.augment_gt_molecular_graph()
        return molecular_graph 

    def __getitem__(self, index, visualization=False):
        while True:
            # Read keypoints
            keypoints = self.dataset["keypoints"].iloc[index]
            
            # Read image
            image_filename = self.dataset["image_filename"].iloc[index]
            image = Image.open(image_filename).convert("RGB") 
            
            # Read graph annotation 
            annotation = self.dataset["annotation"].iloc[index]

            if (image.size[0] != self.config["image_size"][1]) or (image.size[1] != self.config["image_size"][2]):
                # Threshold images
                image = np.array(image, dtype=np.float32)/255
                image[image > 0.8] = 1.
                image[image != 1.] = 0.
                
                # Resize image (768, 768, 3) -> (1024, 1024, 3)
                image = cv2.resize(image, (self.config["image_size"][1], self.config["image_size"][1]), interpolation = cv2.INTER_AREA)

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
            
            # Get molecular graph
            molecular_graph = self.get_molecular_graph(image, keypoints, annotation)
            
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

            return {
                "data": data,
                "images": image
            }
            