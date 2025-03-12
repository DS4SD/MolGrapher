#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import copy
import shutil
import cv2
import ast 
import argparse
import pandas as pd 
from PIL import Image
from pprint import pprint
from tqdm import tqdm 
from time import time
from more_itertools import chunked
from rdkit import Chem
from rdkit.Chem import rdmolfiles
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt

from molgrapher.data_modules.data_module import DataModule
from molgrapher.datasets.dataset_image import ImageDataset
from molgrapher.models.abbreviation_detector import AbbreviationDetectorCPU, AbbreviationDetectorGPU, SpellingCorrector
from molgrapher.models.graph_recognizer import GraphRecognizer, StereochemistryRecognizer
from molgrapher.utils.utils_dataset import get_bonds_sizes
from molgrapher.utils.utils_logging import count_model_parameters

from mol_depict.utils.utils_generation import get_abbreviations_smiles_mapping
from mol_depict.utils.utils_drawing import draw_molecule_rdkit

os.environ["OMP_NUM_THREADS"] = "1" 
cv2.setNumThreads(0)
torch.set_float32_matmul_precision("medium")


class MolgrapherModel:
    def __init__(self, args={}):   
        self.args = {
            "force_cpu": False,
            "force_no_multiprocessing": True, # Disable PaddleOCR multiprocessing for abbreviation detection
            "num_threads_pytorch": 10,
            "num_processes_mp": 10,
            "chunk_size": 200,
            "assign_stereo": True,
            "align_rdkit_output": False,
            "remove_captions": True,
            "save_mol_folder": "",
            "config_dataset_graph_path": os.path.dirname(__file__) + "/../../data/config_dataset_graph_2.json",
            "config_training_graph_path": os.path.dirname(__file__) + "/../../data/config_training_graph.json",
            "config_dataset_keypoint_path": os.path.dirname(__file__) + "/../../data/config_dataset_keypoint.json",
            "config_training_keypoint_path": os.path.dirname(__file__) + "/../../data/config_training_keypoint.json",
            "predict": True,
            "preprocess": True,
            "clean": True,
            "visualize": True,
            "visualize_rdkit": False,            
            "visualize_output_folder_path": os.path.dirname(__file__) + "/../../data/visualization/predictions/default/",
            "visualize_rdkit_output_folder_path": os.path.dirname(__file__) + "/../../data/visualization/predictions/default_rdkit/"
        }

        self.args.update(args)

        print("Arguments:")
        pprint(self.args)

        # Create save folders
        if self.args["visualize"]:
            if self.args["clean"] and (os.path.exists(self.args["visualize_output_folder_path"])):
                shutil.rmtree(self.args["visualize_output_folder_path"])
            if not os.path.exists(self.args["visualize_output_folder_path"]):
                os.makedirs(self.args["visualize_output_folder_path"])
        if self.args["predict"] and (self.args["save_mol_folder"] != ""):
            if self.args["clean"] and (os.path.exists(self.args["save_mol_folder"])):
                shutil.rmtree(self.args["save_mol_folder"])
            if not os.path.exists(self.args["save_mol_folder"]):
                os.makedirs(self.args["save_mol_folder"])

        # Automatically set CPU/GPU device
        if not(self.args["force_cpu"]):
            self.args["force_cpu"] = not(torch.cuda.is_available())
        print(f"PyTorch device: {'gpu' if not(self.args['force_cpu']) else 'cpu'}")

        # Read config file
        with open(self.args["config_dataset_graph_path"]) as file:
            self.config_dataset_graph = json.load(file)
        with open(self.args["config_training_graph_path"]) as file:
            self.config_training_graph = json.load(file)
        with open(self.args["config_dataset_keypoint_path"]) as file:
            self.config_dataset_keypoint = json.load(file)
        with open(self.args["config_training_keypoint_path"]) as file:
            self.config_training_keypoint = json.load(file)

        # Update config
        self.config_dataset_graph["num_processes_mp"] = self.args["num_processes_mp"]
        self.config_dataset_graph["num_threads_pytorch"] = self.args["num_threads_pytorch"]
        self.config_dataset_keypoint["num_processes_mp"] = self.args["num_processes_mp"]
        self.config_dataset_keypoint["num_threads_pytorch"] = self.args["num_threads_pytorch"]
        
        # Set # threads
        torch.set_num_threads(self.config_dataset_graph["num_threads_pytorch"])

        # Read model
        self.model = GraphRecognizer(
            self.config_dataset_keypoint,
            self.config_training_keypoint,
            self.config_dataset_graph,
            self.config_training_graph
        )
        print(f"Keypoint detector number parameters: {round(count_model_parameters(self.model.keypoint_detector)/10**6, 4)} M")
        print(f"Node classifier number parameters: {round(count_model_parameters(self.model.graph_classifier)/10**6, 4)} M")

        # Set up trainer
        if self.args["force_cpu"]:
            self.trainer = pl.Trainer(
                accelerator="cpu",
                precision=self.config_training_graph["precision"],
                logger=False,
            )
        else:
            self.trainer = pl.Trainer(
                accelerator=self.config_training_graph["accelerator"],
                devices=self.config_training_graph["devices"],
                precision=self.config_training_graph["precision"],
                logger=False,
            )

        # Setup abbreviation detector
        if self.args["force_cpu"] or (self.config_training_graph["accelerator"] == "cpu"):
            self.abbreviation_detector = AbbreviationDetectorCPU(
                self.config_dataset_graph, 
                force_cpu=self.args["force_cpu"], 
                force_no_multiprocessing=self.args["force_no_multiprocessing"]
            )
        else:
            self.abbreviation_detector = AbbreviationDetectorGPU(
                self.config_dataset_graph, 
                force_cpu=self.args["force_cpu"], 
                force_no_multiprocessing=self.args["force_no_multiprocessing"]
            )

        # Setup stereochemistry recognizer
        self.stereochemistry_recognizer = StereochemistryRecognizer(self.config_dataset_graph)

        # Set abbreviations list
        with open(os.path.dirname(__file__) + "/../../data/ocr_mapping/ocr_atoms_classes_mapping.json") as file:
            self.ocr_atoms_classes_mapping = json.load(file)
        
        self.abbreviations_smiles_mapping = get_abbreviations_smiles_mapping()
        self.spelling_corrector = SpellingCorrector(self.abbreviations_smiles_mapping)

    def predict_batch(self, _images_paths):
        annotations_batch = []
        for _batch_images_paths in chunked(_images_paths, self.args["chunk_size"]):
            annotations_batch.extend(self.predict(_batch_images_paths))
        return annotations_batch

    def predict(self, images_or_paths):
        if not isinstance(images_or_paths, list):
            images_or_paths = [images_or_paths]

        # Read dataset images
        data_module = DataModule(
            self.config_dataset_graph,
            dataset_class = ImageDataset,
            images_or_paths = images_or_paths,
            force_cpu = self.args["force_cpu"],
            remove_captions = self.args["remove_captions"],
        )
        data_module.setup_images_benchmarks()
        if self.args["preprocess"]:
            print(f"Starting Caption Removal Preprocessing")
            ref_t = time()
            data_module.preprocess()
            print(f"Caption Removal Preprocessing completed in {round(time() - ref_t, 2)}")

        # Get predictions
        print(f"Starting Keypoint Detection + Node Classification")
        ref_t = time()
        predictions_out = self.trainer.predict(
            self.model, dataloaders=data_module.predict_dataloader()
        )
        if (predictions_out == None):
            predictions_out = []
        print(f"Keypoint Detection + Node Classification completed in {round(time() - ref_t, 2)}")

        images_filenames = []
        images_ = []
        predictions = {
            "graphs": [], 
            "keypoints": [], 
            "confidences": []
        }
        for _ in range(len(predictions_out)):
            _prediction = predictions_out.pop(0)
            if _prediction is None:
                continue
            for _elem in _prediction["predictions_batch"]["graphs"]:
                predictions["graphs"].append(_elem)
            for _elem in _prediction["predictions_batch"]["keypoints_batch"]:
                predictions["keypoints"].append(_elem)
            for _elem in _prediction["predictions_batch"]["confidences"]:
                predictions["confidences"].append(_elem)
            for _elem in _prediction["batch"]["images_filenames"]:
                images_filenames.append(_elem)
            for _elem in _prediction["batch"]["images"]:
                images_.append(_elem)

        scaling_factor = self.config_dataset_keypoint["image_size"][1]//self.config_dataset_keypoint["mask_size"][1]

        # Compute bond size
        bonds_sizes = get_bonds_sizes(predictions["keypoints"], scaling_factor)

        # Recognize abbreviations
        print(f"Starting Abbreviation Recognition")
        ref_t = time()
        abbreviations_list = self.abbreviation_detector.mp_run(images_filenames, predictions["graphs"], bonds_sizes, filter=False)
        abbreviations_list_ocr = copy.deepcopy(abbreviations_list)
        print(f"Abbreviation Recognition completed in {round(time() - ref_t, 2)}")

        # Recognize stereochemistry
        if self.args["assign_stereo"]:
            print(f"Starting Stereochemistry Recognition")
            ref_t = time()
            predictions["graphs"] = self.stereochemistry_recognizer(images_, predictions["graphs"], bonds_sizes)
            print(f"Stereochemistry Recognition completed in {round(time() - ref_t, 2)}")

        # Create RDKit graph        
        print("Starting Graph creation")
        ref_t = time()
        predicted_molecules = []
        for abbreviations, graph, p in zip(abbreviations_list, predictions["graphs"], images_filenames):
            predicted_molecule = graph.to_rdkit(
                abbreviations,
                self.abbreviations_smiles_mapping,
                self.ocr_atoms_classes_mapping,
                self.spelling_corrector,
                assign_stereo=self.args["assign_stereo"],
                align_rdkit_output=self.args["align_rdkit_output"],
                postprocessing_flags = {}
            )
            predicted_molecules.append(predicted_molecule)
    
        print(f"Graph creation completed in {round(time() - ref_t, 2)}")
        predictions["molecules"] = predicted_molecules

        # Convert to SMILES and set confidence
        predictions["smiles"] = []
        for i, (predicted_molecule, image_filename) in enumerate(zip(predictions["molecules"], images_filenames)):
            if self.args["save_mol_folder"] != "":
                molecule_path = self.args["save_mol_folder"] + image_filename.split("/")[-1][:-4].replace("_preprocessed", "") + ".mol"
                rdmolfiles.MolToMolFile(
                    predicted_molecule, 
                    molecule_path, 
                    kekulize = False 
                )
            smiles = Chem.MolToSmiles(predicted_molecule)
            if smiles:
                predictions["smiles"].append(smiles)
                if smiles == "C":
                    predictions["confidences"][i] = 0
            else:
                predictions["smiles"].append(None)
                predictions["confidences"][i] = 0
                print("The molecule can not be converted to a valid SMILES")

        # Save annotations
        annotations = []
        for predicted_smiles, confidence, image_filename, abbreviations, abbreviations_ocr in zip(predictions["smiles"], predictions["confidences"], images_filenames, abbreviations_list, abbreviations_list_ocr):
            if predicted_smiles is not None:
                if abbreviations != []:
                    abbreviations_texts = [abbreviation["text"] for abbreviation in abbreviations]
                else:
                    abbreviations_texts = []
                if abbreviations_ocr != []:
                    abbreviations_ocr_texts = [abbreviation["text"] for abbreviation in abbreviations_ocr]
                else:
                    abbreviations_ocr_texts = []

                annotation = {
                    "smi": predicted_smiles,
                    "abbreviations": abbreviations_texts,
                    "abbreviations_ocr": abbreviations_ocr_texts,
                    "conf": confidence,
                    "file-info": {
                        "filename": image_filename, 
                        "image_nbr": 1
                    },
                    "annotator": {
                        "version": "1.0.0",
                        "program": "MolGrapher"
                    }
                }  
                annotations.append(annotation)
            
            if self.args["save_mol_folder"] != "":
                annotation_filename = self.args["save_mol_folder"] + "smiles.jsonl"            
                with open(annotation_filename, "a") as f:
                    json.dump(annotation, f)
                    f.write('\n')

        if self.args["save_mol_folder"] != "":    
            print("Annotation:")
            print(pd.read_json(path_or_buf = annotation_filename, lines = True))

        # Visualize predictions
        if self.args["visualize"]:
            for image_filename, image, graph, keypoints, molecule in tqdm(zip(images_filenames, images_, predictions["graphs"], predictions["keypoints"], predictions["molecules"]), total=len(images_filenames)):
                smiles = Chem.MolToSmiles(molecule)
                if smiles != "C":
                    figure, axis = plt.subplots(1, 3, figsize=(20, 10))
                else: 
                    figure, axis = plt.subplots(1, 2, figsize=(20, 10))
                axis[0].imshow(image.permute(1, 2, 0))
            
                axis[0].scatter(
                    [(keypoint[0]*scaling_factor + scaling_factor//2) for keypoint in keypoints], 
                    [(keypoint[1]*scaling_factor + scaling_factor//2) for keypoint in keypoints], 
                    color = "red",
                    alpha = 0.5
                )

                graph.display_data_nodes_only(axis=axis[1])
                
                if smiles != "C":
                    image = draw_molecule_rdkit(
                        smiles = smiles,
                        molecule = molecule,
                        augmentations = False,
                    )
                    if image is not None:
                        axis[2].imshow(image.permute(1, 2, 0))
                print(f"{self.args['visualize_output_folder_path']}/{image_filename.split('/')[-1]}")
                plt.savefig(f"{self.args['visualize_output_folder_path']}/{image_filename.split('/')[-1]}")
                plt.close()

        if self.args["visualize_rdkit"]:
            for image_filename in self.args["input_images_paths"]:
                molecule_path = self.args["save_mol_folder"] + image_filename.split("/")[-1][:-4].replace("_preprocessed", "") + ".mol"
                if os.path.exists(molecule_path):
                    print(molecule_path)
                    image = Image.open(image_filename).convert("RGB")
                    figure, axis = plt.subplots(1, 2, figsize=(20, 10))
                    axis[0].imshow(image)
                    molecule = rdmolfiles.MolFromMolFile(molecule_path, sanitize = False)
                    image = draw_molecule_rdkit(
                        smiles = Chem.MolToSmiles(molecule),
                        molecule = molecule,
                        augmentations = False,
                    )
                    if image is not None:
                        axis[1].imshow(image.permute(1, 2, 0))
                    plt.savefig(f"{self.args['visualize_rdkit_output_folder_path']}/{image_filename.split('/')[-1]}")
                plt.close()

        return annotations
