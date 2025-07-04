#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import pytorch_lightning as pl
from torch_geometric.data import Batch
from time import time
import numpy as np
import json 
from math import *
import cv2
import subprocess
from huggingface_hub import snapshot_download

from molgrapher.models.graph_constructor import GraphConstructor
from molgrapher.models.keypoint_detector import KeypointDetector
from molgrapher.models.graph_classifier import GraphClassifier


class GraphRecognizer(pl.LightningModule):
    def __init__(
        self, 
        config_dataset_keypoint, 
        config_training_keypoint, 
        config_dataset_graph, 
        config_training_graph, 
        config_model_graph=None, 
        keypoint_detector_model_path="",
        graph_classifier_model_path=""
    ):
        super().__init__()
        self.config_dataset_graph = config_dataset_graph
        self.config_dataset_keypoint = config_dataset_keypoint

        if keypoint_detector_model_path == "":
            keypoint_detector_model_path = os.path.dirname(__file__) + f"/../../data/models/keypoint_detector/kd_model.ckpt" 
            if not(os.path.exists(keypoint_detector_model_path)):
                print("Downloading keypoint detector model...")
                subprocess.run([
                    "wget",
                    "https://huggingface.co/ds4sd/MolGrapher/resolve/main/models/keypoint_detector/kd_model.ckpt",
                    "-P",
                    "./data/models/keypoint_detector/"
                ], check=True)
        if graph_classifier_model_path == "":
            graph_classifier_model_path = os.path.dirname(__file__) + f"/../../data/models/graph_classifier/{config_model_graph['node_classifier_variant']}.ckpt"
            if not(os.path.exists(graph_classifier_model_path)):
                print("Downloading node classifier model...")
                subprocess.run([
                    "wget", 
                    f"https://huggingface.co/ds4sd/MolGrapher/resolve/main/models/graph_classifier/{config_model_graph['node_classifier_variant']}.ckpt", 
                    "-P", 
                    "./data/models/graph_classifier/"
                ], check=True)

        self.keypoint_detector = KeypointDetector.load_from_checkpoint(
            keypoint_detector_model_path, 
            config_dataset = config_dataset_keypoint,
            config_training = config_training_keypoint,
            map_location = self.device
        )
        
        self.graph_classifier = GraphClassifier.load_from_checkpoint(
            graph_classifier_model_path, 
            config_dataset = config_dataset_graph, 
            config_training = config_training_graph,
            gcn_on = config_model_graph["gcn_on"],
            map_location = self.device
        )
        
        print("Selected keypoint detector model: ", keypoint_detector_model_path)
        print("Selected node classifier model: ", graph_classifier_model_path)

    def predict(self, batch, use_gt_structure=False, verbose=False, return_all_steps=False):
        if verbose:
            print("--------------------------------------------------")
        ref_t = time()

        images = batch["images"]
        self.keypoint_detector.eval()
        with torch.no_grad():
            if use_gt_structure:
                # Use ground truth structure for debugging
                keypoints_batch = batch["keypoints_batch"]
            else:
                # Keypoints detection
                keypoints_predictions = self.keypoint_detector.predict(images)
                keypoints_batch, heatmaps_batch = keypoints_predictions["keypoints_batch"], keypoints_predictions["heatmaps_batch"]

            if verbose:
                print(f"Keypoint detection completed in: {round(time() - ref_t, 2)}")
            ref_t = time()

            # Dense graph structure construction
            graphs_structures = []
            for index, (keypoints, image) in enumerate(zip(keypoints_batch, images)): 
                # Could be parallelized
                graph_constructor = GraphConstructor(
                    keypoints,
                    image, 
                    self.config_dataset_keypoint, 
                    self.config_dataset_graph,
                    discarded_bond_length_factor = 1.75
                )
                molecular_graph = graph_constructor.get_molecular_graph()
                molecular_graph.set_nodes_edges_nodes_only()
                graph_structure = molecular_graph.to_torch_nodes_only()
                graphs_structures.append(graph_structure)

            graphs_structures = Batch.from_data_list(graphs_structures).to(self.device)
            
            if verbose:
                print(f"Dense graph construction completed in: {round(time() - ref_t, 2)}")
            ref_t = time()

        self.graph_classifier.eval()
        with torch.no_grad():
            batch = {
                "images": images,
                "data": graphs_structures
            }
            # Nodes and edges classification
            predictions, confidences_batch = self.graph_classifier.predict(batch)
            if verbose:
                print(f"Nodes classification completed in {round(time() - ref_t, 3)}")
            ref_t = time()
            if use_gt_structure:
                return {
                    "graphs": predictions,
                    "keypoints_batch": keypoints_batch,
                    "confidences": confidences_batch
                }
            if return_all_steps:
                return {
                    "graphs": predictions,
                    "keypoints_batch": keypoints_batch,
                    "heatmaps_batch": heatmaps_batch,
                    "confidences": confidences_batch
                }
            return {
                "graphs": predictions,
                "keypoints_batch": keypoints_batch,
                "confidences": confidences_batch
            }
    
    def predict_step(self, batch, batch_idx, drop_last = False):
        torch.set_num_threads(self.config_dataset_graph["num_threads_pytorch"])
        if drop_last and (len(batch["images"]) < self.config_dataset_graph["batch_size"]):
            print(f"The proposed batch size: {len(batch['images'])} is smaller than the configurated batch size: {self.config_dataset_graph['batch_size']}")
            return None

        return {
            "predictions_batch": self.predict(batch),
            "batch": batch
        }

class StereochemistryRecognizer():
    def __init__(self, config):
        symbols_classes_atoms = json.load(open(os.path.dirname(__file__) + f"/../../data/vocabularies/vocabulary_atoms_{config['nb_atoms_classes']}.json"))
        types_classes_bonds = json.load(open(os.path.dirname(__file__) + f"/../../data/vocabularies/vocabulary_bonds_{config['nb_bonds_classes']}.json"))
        self.atoms_classes_symbols = {v: k for k,v in symbols_classes_atoms.items()}
        self.bonds_classes_types = {v: k for k,v in types_classes_bonds.items()}
        
    def get_bond_direction(self, image, bond, atoms, bond_length):
        bond_positions = [
            atoms[bond["index"][0]]["position"], 
            atoms[bond["index"][1]]["position"]
        ] 
        test_window_size = (
            int(0.3*bond_length), 
            int(0.3*bond_length)
        )
        center_up = (
            (bond_positions[0][0]*3 + bond_positions[1][0])/4, 
            (bond_positions[0][1]*3 + bond_positions[1][1])/4
        )
        center_down = (
            (bond_positions[0][0] + bond_positions[1][0]*3)/4, 
            (bond_positions[0][1] + bond_positions[1][1]*3)/4
        )
        delta_x = bond_positions[0][0] - bond_positions[1][0]
        delta_y = bond_positions[0][1] - bond_positions[1][1]
        if delta_x == 0:
            angle = int((np.pi/2)*(180/np.pi))
        else:
            angle = int(atan(delta_y/delta_x)*(180/np.pi))
        rectangle_up = [center_up, test_window_size, angle]
        rectangle_down = [center_down, test_window_size, angle]

        mask = np.zeros((1024, 1024, 3), dtype=float)
        mask_up = torch.tensor(cv2.fillPoly(mask, pts =[np.int0(cv2.boxPoints(rectangle_up))], color=(1, 1, 1))).permute(2, 0, 1)
        masked_image_up = (1 - image)*mask_up

        mask = np.zeros((1024, 1024, 3), dtype=float)
        mask_down = torch.tensor(cv2.fillPoly(mask, pts =[np.int0(cv2.boxPoints(rectangle_down))], color=(1, 1, 1))).permute(2, 0, 1)
        masked_image_down = (1 - image)*mask_down

        if (torch.sum(masked_image_up == True).item()) > (torch.sum(masked_image_down == True).item()):
            return self.bonds_classes_types[bond["class"]] + "-UP"
        else:
            return self.bonds_classes_types[bond["class"]] + "-DOWN"

    def __call__(self, images, graphs, bonds_sizes):
        for bond_size, image, graph in zip(bonds_sizes, images, graphs):
            for bond in graph.bonds:
                if (self.bonds_classes_types[bond["class"]] == "DASHED") or (self.bonds_classes_types[bond["class"]] == "SOLID"):
                    bond["class"] = self.get_bond_direction(image, bond, graph.atoms, bond_size)
        return graphs
        