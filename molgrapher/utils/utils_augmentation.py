#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python
import json 
import os 
import math 
import random 

# Mathematics
import numpy as np 

# Images
import cv2
import albumentations as albu
from albumentations.augmentations.geometric.transforms import Perspective, PiecewiseAffine, Affine
from albumentations.augmentations.blur.transforms import GaussianBlur
from albumentations.augmentations.geometric.resize import Resize 

# Modules
from mol_depict.utils.image_transformation import RandomCaption, RandomLines, PepperPatches


def get_transforms_dict(config):
    transforms = {}

    # Soft
    transforms_list = []
    transforms_list.append(albu.ShiftScaleRotate(
        rotate_limit=10, 
        scale_limit=(-0.2, 0), 
        shift_limit=(-0.01, 0.01), 
        p=0.9, 
        border_mode=cv2.BORDER_CONSTANT, 
        value=(1., 1., 1.), 
        interpolation=cv2.INTER_NEAREST
    )) 
    transforms_list.append(PepperPatches(p=0.1))  
    transforms_list.append(
        albu.OneOf([
            albu.Downscale(
                scale_min=0.35, 
                scale_max=0.85, 
                p=0.5, 
                interpolation={"downscale": cv2.INTER_NEAREST, "upscale": cv2.INTER_NEAREST}
            ),
            GaussianBlur(
                blur_limit=21, 
                p=0.25
            ),
            albu.Compose([
                Affine(
                    shear=({"x": (-20, 20), "y": (-20, 20)}),
                    p=1,
                    fit_output=True,
                    cval=(1, 1, 1)
                ),
                Resize(
                    config["image_size"][1],
                    config["image_size"][2]
                )
            ], p=0.25)
        ], p=0.8)
    ) 
    transforms["soft"] = albu.Compose(transforms_list, keypoint_params=albu.KeypointParams(format='xy', remove_invisible=True))

    # Hard
    transforms_list = []
    transforms_list.append(albu.ShiftScaleRotate(
        rotate_limit=10, 
        scale_limit=(-0.2, 0), 
        shift_limit=(-0.01, 0.01), 
        p=0.9, 
        border_mode=cv2.BORDER_CONSTANT, 
        value=(1., 1., 1.), 
        interpolation=cv2.INTER_NEAREST
    )) 
    transforms_list.append(RandomCaption(p=0.1))
    transforms_list.append(RandomLines(p=0.05)) 
    #transforms_list.append(Perspective(p=0.05, scale=(0.05, 0.15), fit_output=True, pad_val=(1, 1))) # Performance issue
    #transforms_list.append(PiecewiseAffine(p=1)) # Performance issue
    transforms_list.append(PepperPatches(p=0.1))  
    transforms_list.append(
        albu.OneOf([
            albu.Downscale(
                scale_min=0.5, 
                scale_max=0.85, 
                p=0.5, 
                interpolation={"downscale": cv2.INTER_NEAREST, "upscale": cv2.INTER_NEAREST}
            ),
            GaussianBlur(
                blur_limit=11,
                p=0.25
            ),
            albu.Compose([
                Affine(
                    shear=({"x": (-20, 20), "y": (-20, 20)}),
                    p=1,
                    fit_output=True,
                    cval=(1, 1, 1)
                ),
                Resize(
                    config["image_size"][1],
                    config["image_size"][2]
                )
            ], p=0.25)
        ], p=0.8)
    ) 
    transforms["hard"] = albu.Compose(transforms_list, keypoint_params=albu.KeypointParams(format='xy', remove_invisible=True))

    # Extreme
    transforms_list = []
    transforms_list.append(albu.ShiftScaleRotate( 
        rotate_limit=360, 
        scale_limit=(-0.5, 0), 
        shift_limit=(-0.01, 0.01), 
        p=0.9, 
        border_mode=cv2.BORDER_CONSTANT, 
        value=(1., 1., 1.), 
        interpolation=cv2.INTER_NEAREST
    )) 
    transforms_list.append(RandomCaption(p=0.3)) 
    #transforms_list.append(RandomLines(p=0.20)) 
    transforms_list.append(PepperPatches(p=0.30))  
    transforms_list.append(
        albu.OneOf([
            albu.Downscale(
                scale_min=0.25, 
                scale_max=0.85, 
                p=0.5, 
                interpolation={"downscale": cv2.INTER_NEAREST, "upscale": cv2.INTER_NEAREST}
            ),
            GaussianBlur(
                blur_limit=51,
                p=0.25
            ),
            albu.Compose([
                Affine(
                    shear=({"x": (-60, 60), "y": (-60, 60)}),
                    p=1,
                    fit_output=True,
                    cval=(1, 1, 1)
                ),
                Resize(
                    config["image_size"][1],
                    config["image_size"][2]
                )
            ], p=0.25)
        ], p=0.8)
    ) 
    transforms["extreme"] = albu.Compose(transforms_list, keypoint_params=albu.KeypointParams(format='xy', remove_invisible=True))
    
    # Debug
    transforms["debug"] = albu.Compose([], keypoint_params=albu.KeypointParams(format='xy', remove_invisible=True)) 
    return transforms


class GraphTransformer():
    def __init__(self, config, keypoints_shift_limit, decoy_keypoint_shift_limit, decoy_atom_population_density, keypoints_only=False):
        self.config = config
        self.keypoints_shift_limit = keypoints_shift_limit
        self.decoy_keypoint_shift_limit = decoy_keypoint_shift_limit
        self.decoy_atom_population_density = decoy_atom_population_density

        if not keypoints_only:
            self.symbols_classes_atoms = json.load(open(os.path.dirname(__file__) + f"/../../data/vocabularies/vocabulary_atoms_{config['nb_atoms_classes']}.json"))
            self.types_classes_bonds = json.load(open(os.path.dirname(__file__) + f"/../../data/vocabularies/vocabulary_bonds_{config['nb_bonds_classes']}.json"))

    def shift_keypoints_positions(self, keypoints, shift_window):
        shifted_keypoints = []
        for keypoint in keypoints:
            center_x, center_y = keypoint
            radius = (shift_window[1]) * math.sqrt(random.random())
            theta = random.random() * 2 * np.pi
            keypoint = [
                max(min(int(center_x + radius * np.cos(theta)), self.config["image_size"][1] - 1), shift_window[0]), 
                max(min(int(center_y + radius * np.sin(theta)), self.config["image_size"][1] - 1), shift_window[0])
            ]
            shifted_keypoints.append(keypoint)
        return shifted_keypoints

    def shift_atoms_positions(self, graph):
        shift_window = (self.keypoints_shift_limit[0]*graph.bond_size, self.keypoints_shift_limit[1]*graph.bond_size)

        keypoints = []
        for atom in graph.atoms:
            keypoints.append([atom["position"][0], atom["position"][1]])

        keypoints = self.shift_keypoints_positions(keypoints, shift_window)
        for index, keypoint in enumerate(keypoints):
            if graph.atoms[index]:
                graph.atoms[index]["position"] = keypoint 

        return graph

    def add_decoy_atom(self, graph):
        shift_window = (self.decoy_keypoint_shift_limit[0]*graph.bond_size, self.decoy_keypoint_shift_limit[1]*graph.bond_size)

        # Get atoms neighbors
        atoms_neighboring_bonds_indices = defaultdict(list)
        for bond_index, bond in enumerate(graph.bonds):
            atoms_neighboring_bonds_indices[bond["index"][0]].append(bond_index)
            atoms_neighboring_bonds_indices[bond["index"][1]].append(bond_index)

        new_atoms = graph.atoms.copy()
        new_atom_index = len(graph.atoms) 

        # Select a random atom to duplication
        atom_index = random.choice(list(range(len(graph.atoms))))
        atom = graph.atoms[atom_index]
        
        # Add decoy atom with shifted position
        keypoints = self.shift_keypoints_positions([[atom["position"][0], atom["position"][1]]], shift_window) 
        new_atom_position = keypoints[0]
        new_atoms.append({
            "index": new_atom_index,
            "class": self.symbols_classes_atoms["Decoy"],
            "position": new_atom_position,
            "type": 1
        })
        
        # Connect decoy atom to original neighbors
        for neighbor_bond_index in atoms_neighboring_bonds_indices[atom_index]:
            if atom_index == graph.bonds[neighbor_bond_index]["index"][0]:
                neighbor_atom_index = graph.bonds[neighbor_bond_index]["index"][1]
            elif atom_index == graph.bonds[neighbor_bond_index]["index"][1]:
                neighbor_atom_index = graph.bonds[neighbor_bond_index]["index"][0]
            neighbor_atom_position =  graph.atoms[neighbor_atom_index]["position"]
            
            bond = {
                "index": [new_atom_index, neighbor_atom_index],
                "class": self.types_classes_bonds["Decoy"],
                "position": [
                    (new_atom_position[0] + neighbor_atom_position[0])//2, 
                    (new_atom_position[1] + neighbor_atom_position[1])//2
                ],
                "type": 0
            }
            graph.bonds.append(bond) 

        # Add bond in-between original atom and decoy atom
        atom_position =  graph.atoms[atom_index]["position"]
        graph.bonds.append({
            "index": [new_atom_index, atom_index],
            "class": self.types_classes_bonds["Decoy"],
            "position": [
                (new_atom_position[0] + atom_position[0])//2, 
                (new_atom_position[1] + atom_position[1])//2
            ],
            "type": 0
        }) 

        graph.atoms = new_atoms
        return graph
        
    def augment(self, graph):
        graph = self.shift_atoms_positions(graph)
        if self.decoy_atom_population_density > 0:
            if random.random() < self.decoy_atom_population_density:
                graph = self.add_decoy_atom(graph)
        return graph