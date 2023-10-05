#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
from math import *
import torch
import json 
import os
from torchvision.transforms import functional

from molgrapher.utils.utils_graph import MolecularGraph


class GraphConstructor:
    def __init__(
        self, 
        keypoints, 
        image, 
        config_dataset_keypoint, 
        config_dataset_graph,
        gt_molecular_graph=None, 
        discarded_bond_length_factor=2.5, 
        restricted_proposals=True
    ):
        self.config_dataset_keypoint = config_dataset_keypoint
        self.config_dataset_graph = config_dataset_graph
        self.max_number_neighbors = 6
        self.empty_test_window = [0.3, 0.3]
        self.discarded_bond_length_factor = discarded_bond_length_factor 
        self.restricted_proposals = restricted_proposals

        self.symbols_classes_atoms = json.load(open(os.path.dirname(__file__) + f"/../../data/vocabularies/vocabulary_atoms_{self.config_dataset_graph['nb_atoms_classes']}.json"))
        self.types_classes_bonds = json.load(open(os.path.dirname(__file__) + f"/../../data/vocabularies/vocabulary_bonds_{self.config_dataset_graph['nb_bonds_classes']}.json"))

        self.image = image
        self.gt_molecular_graph = gt_molecular_graph
        
        if "mask_size" in self.config_dataset_keypoint:
            scaling_factor = self.config_dataset_keypoint["image_size"][1]/self.config_dataset_keypoint["mask_size"][1]
            self.keypoints = [
                [int(keypoint[0]*scaling_factor + (scaling_factor//2)), 
                int(keypoint[1]*scaling_factor + (scaling_factor//2))] for keypoint in keypoints
            ]

        else:
            self.keypoints = keypoints

        self.bonds = []
        self.bonds_candidates = []
        self.set_bonds(self.restricted_proposals)

    def is_on_right_side(self, x, y, xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        a = float(y1 - y0)
        b = float(x0 - x1)
        c = - a*x0 - b*y0
        return a*x + b*y + c >= 0

    def test_point(self, x, y, vertices):
        num_vert = len(vertices)
        is_right = [self.is_on_right_side(x, y, vertices[i], vertices[(i + 1) % num_vert]) for i in range(num_vert)]
        all_left = not any(is_right)
        all_right = all(is_right)
        return all_left or all_right

    def _is_obstructed(self, bond):
        """
        It is acceptable to check only the central location of the proposed bond because almost everytime bonds are aligned, they are only two.
        """
        bond_positions = [
            [self.keypoints[bond[0]][0], self.keypoints[bond[0]][1]], 
            [self.keypoints[bond[1]][0], self.keypoints[bond[1]][1]]
        ] 
        bond_length = np.sqrt((bond_positions[0][0] - bond_positions[1][0])**2 + (bond_positions[0][1] - bond_positions[1][1])**2)
        test_window_size = (
            int(bond_length*0.8),
            int(bond_length*0.2),  
        )
        center = (
            (bond_positions[0][0] + bond_positions[1][0])//2, 
            (bond_positions[0][1] + bond_positions[1][1])//2 
        )

        delta_x = bond_positions[0][0] - bond_positions[1][0]
        delta_y = bond_positions[0][1] - bond_positions[1][1]
        if delta_x == 0:
            angle = math.degrees(np.pi/2)
        else:
            angle = math.degrees(atan(delta_y/delta_x))

        rectangle = [center, test_window_size, angle]
        test_window = cv2.boxPoints(rectangle).tolist()

        for keypoint in self.keypoints:
            if self.test_point(keypoint[0], keypoint[1], test_window):
                return True

        return False

    def _is_too_long(self, bond):
        bond_positions = [
            [self.keypoints[bond[0]][0], self.keypoints[bond[0]][1]], 
            [self.keypoints[bond[1]][0], self.keypoints[bond[1]][1]]
        ] 
        length = np.sqrt((bond_positions[0][0] - bond_positions[1][0])**2 + (bond_positions[0][1] - bond_positions[1][1])**2)
        return length > self.discarded_bond_length_factor*self.bond_length

    def _is_empty_fast(self, bond):
        bond_positions = [
            [self.keypoints[bond[0]][0], self.keypoints[bond[0]][1]], 
            [self.keypoints[bond[1]][0], self.keypoints[bond[1]][1]]
        ] 
        
        center = (
            (bond_positions[0][0] + bond_positions[1][0])//2, 
            (bond_positions[0][1] + bond_positions[1][1])//2
        )
        
        image_crop = functional.crop(
            self.image, 
            top = center[1] - (self.test_window_size[1]//2),
            left = center[0] - (self.test_window_size[0]//2),
            height = self.test_window_size[1],
            width = self.test_window_size[0]
        )
 
        return (image_crop != 1.).sum() < 1

    def _is_empty(self, bond):
        bond_positions = [
            [self.keypoints[bond[0]][0], self.keypoints[bond[0]][1]], 
            [self.keypoints[bond[1]][0], self.keypoints[bond[1]][1]]
        ] 
        test_window_size = (
            int(self.empty_test_window[0]*self.bond_length), 
            int(self.empty_test_window[1]*self.bond_length)
        )
        center = (
            (bond_positions[0][0] + bond_positions[1][0])/2, 
            (bond_positions[0][1] + bond_positions[1][1])/2
        )
        delta_x = bond_positions[0][0] - bond_positions[1][0]
        delta_y = bond_positions[0][1] - bond_positions[1][1]
        if delta_x == 0:
            angle = int((np.pi/2)*(180/np.pi))
        else:
            angle = int(atan(delta_y/delta_x)*(180/np.pi))
        rectangle = [center, test_window_size, angle]

        mask = np.zeros((self.config_dataset_keypoint["image_size"][1], self.config_dataset_keypoint["image_size"][2], self.config_dataset_keypoint["image_size"][0]), dtype=np.float)
        mask = cv2.fillPoly(mask, pts =[np.int0(cv2.boxPoints(rectangle))], color=(1, 1, 1))
        mask = torch.tensor(mask).permute(2, 0, 1)
        
        masked_image = (1 - self.image)*mask
        return torch.sum(masked_image == True).item() < 1
        
    def set_candidates_bonds(self):
        """
        Each atom have at least 6 bonds proposals, and possibly more. 
        """
        min_distances = []
        for index_keypoint_query, keypoint_query in enumerate(self.keypoints):
            distances = []
            indices_keypoints_keys = []
            for index_keypoint_key, keypoint_key in enumerate(self.keypoints):
                # Candidate bonds are not duplicated
                if index_keypoint_key < index_keypoint_query:
                    distance = np.sqrt((keypoint_query[0] - keypoint_key[0])**2 + (keypoint_query[1] - keypoint_key[1])**2)
                    distances.append(distance)
                    indices_keypoints_keys.append(index_keypoint_key)

            if len(distances) > 0:
                min_distances.append(min(distances))

                indices_keypoint_neighbors = [
                    index_keypoint for (distance, index_keypoint) in sorted(
                                                                            zip(distances, indices_keypoints_keys), 
                                                                            key = lambda x: x[0]
                                                                        )[:self.max_number_neighbors]
                ]
                
                for index_keypoint_neighbor in indices_keypoint_neighbors:
                    self.bonds_candidates.append([index_keypoint_query, index_keypoint_neighbor])

        if len(min_distances) > 0:
            # Upper bound estimate of the bond length, as the 75th percentile of minimal distances
            self.bond_length = np.percentile(min_distances, 75) 

            self.test_window_size = (
                max(int(self.empty_test_window[0]*self.bond_length), 15), 
                max(int(self.empty_test_window[1]*self.bond_length), 15)
            )

    def set_bonds(self, restricted_proposals):
        self.set_candidates_bonds()
        for bond_candidate in self.bonds_candidates:
            if restricted_proposals and self._is_obstructed(bond_candidate):
                continue

            if restricted_proposals and self._is_empty_fast(bond_candidate):
                continue
            
            if restricted_proposals and self._is_too_long(bond_candidate):
                continue
      
            self.bonds.append(bond_candidate)

    def get_molecular_graph(self):
        molecular_graph = MolecularGraph(self.config_dataset_graph)

        for index, keypoint in enumerate(self.keypoints):
            molecular_graph.atoms.append({
                "index": index,
                "class": self.symbols_classes_atoms["None"],
                "position": keypoint,
                "type": 1
            })
        
        for bond in self.bonds:
            molecular_graph.bonds.append({
                "index": bond,
                "class": self.types_classes_bonds["None"],
                "position": [
                    (self.keypoints[bond[0]][0] + self.keypoints[bond[1]][0])/2, 
                    (self.keypoints[bond[0]][1] + self.keypoints[bond[1]][1])/2
                ],
                "type": 0
            })
        return molecular_graph
       
    def augment_gt_molecular_graph(self):
        gt_bonds_indices = [set(bond["index"]) for bond in self.gt_molecular_graph.bonds]
        for bond in self.bonds:
            if set(bond) not in gt_bonds_indices:
                self.gt_molecular_graph.bonds.append({
                    "index": bond,
                    "class": self.types_classes_bonds["None"],
                    "position": [
                        (self.keypoints[bond[0]][0] + self.keypoints[bond[1]][0])/2, 
                        (self.keypoints[bond[0]][1] + self.keypoints[bond[1]][1])/2
                    ],
                    "type": 0
                })
        return self.gt_molecular_graph