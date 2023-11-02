#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os import path 
import math
import glob
import json
import random
from tqdm import tqdm
from pprint import pprint
import multiprocessing
from rdkit import Chem
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch_geometric
import torch
from torchvision.transforms import functional
from PIL import Image
import cv2
from pytorch_lightning.trainer.states import TrainerFn
import logging
logger = logging.getLogger("ocsr_logger")

from mol_depict.utils.utils_drawing import draw_molecule_keypoints_rdkit
from mol_depict.molfile_parser.label_molfile import LabelMolFile
from mol_depict.molfile_parser.image_registration import get_affine_transformation
from mol_depict.utils.utils_image import transform_png_image, resize_image
try:
    from annotator.labeled_data_utils import LabeledData
except Exception as e: 
    print(f"CEDe not imported: {str(e)}")


class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        config = None, 
        dataset_class = None, 
        mode = 'train', 
        force_precompute = False, 
        return_images_filenames = False, 
        images_folder_path = None, 
        images_paths = None,
        clean_only = False,
        dataset_evaluate = False,
        dataset_predict = True,
        force_cpu = False,
        taa_step = None,
        remove_captions = True
    ):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.benchmarks_datasets = None
        self.synthetic_save_path = os.path.dirname(__file__) + "/../../data/synthetic_images_molecules_keypoints/keypoints_images_filenames_" + self.config["experiment_name"] + ".json"
        self.dataset_class = dataset_class
        self.mode = mode
        self.force_precompute = force_precompute
        self.return_images_filenames = return_images_filenames
        self.images_folder_path = images_folder_path
        self.images_paths = images_paths
        self.clean_only = clean_only
        self.dataset_evaluate = dataset_evaluate
        self.dataset_predict = dataset_predict
        self.force_cpu = force_cpu
        self.taa_step = taa_step
        self.remove_captions = remove_captions

        print("Data module configuration:")
        pprint(self.config)

    def _draw_process(self, images_filenames, smiles_list):
        keypoints_process = []
        for smiles, image_path in tqdm(zip(smiles_list, images_filenames), total=len(smiles_list)):
            logger.info(smiles)
            try:
                keypoints = draw_molecule_keypoints_rdkit(
                        smiles, 
                        path = image_path, 
                        save_molecule = True, 
                        augmentations = True,
                        fake_molecule = True
                )[1]
            except:
                print(f"ERROR smiles: {smiles}. Trying again...")
                keypoints = draw_molecule_keypoints_rdkit(
                        smiles, 
                        path = image_path, 
                        save_molecule = True, 
                        augmentations = True,
                        fake_molecule = True
                )[1]
            keypoints_process.append(keypoints)
       
        return keypoints_process

    def _draw_process_star(self, images_filenames_smiles_list):
        return self._draw_process(*images_filenames_smiles_list)

    def precompute_keypoints_synthetic(self):
        print("Precompute dataset")
        
        if self.config["nb_sample"] > 1500000:
            print("Max number of samples is 1500000")
            return

        # Note: CXSMILES and SMILES with multiple compounds were removed from datasets.
        dataset_triple_bonds_len = sum(1 for line in open(os.path.dirname(__file__) + "/../../data/smiles/experiment-001_triple_bonds.csv")) - 1 
        if self.config["nb_sample"] < dataset_triple_bonds_len:
            smiles_list = list(pd.read_csv(
                os.path.dirname(__file__) + "/../../data/smiles/experiment-001_triple_bonds.csv", 
                skiprows = sorted(random.sample(range(1, dataset_triple_bonds_len + 1), dataset_triple_bonds_len - self.config["nb_sample"])) 
            )["isosmiles"])
        else:
            nb_sample = self.config["nb_sample"] - dataset_triple_bonds_len
            dataset_len = sum(1 for line in open(os.path.dirname(__file__) + "/../../data/smiles/experiment-002.csv")) - 1 
            smiles_list_triple_bonds = list(pd.read_csv(
                os.path.dirname(__file__) + "/../../data/smiles/experiment-001_triple_bonds.csv"
            )["isosmiles"])
            smiles_list = list(pd.read_csv(
                os.path.dirname(__file__) + "/../../data/smiles/experiment-002.csv", 
                skiprows = sorted(random.sample(range(1, dataset_len + 1), dataset_len - nb_sample)) 
            )["isosmiles"])
            smiles_list = smiles_list_triple_bonds + smiles_list

        # Iterate multiple times on the same molecules
        smiles_list = smiles_list*self.config["nb_duplicates"]

        # Debug
        #smiles_list = [smiles_list[40020]]*100
        #smiles_list = ["CC1=C(C=CC(=N1)C#CC(C)(C)C(=O)OCC[S+](C)C)N"]*100

        images_folder_path = os.path.dirname(__file__) + "/../../data/synthetic_images_molecules_keypoints/keypoints_images_filenames_" + self.config["experiment_name"] + "/"
        if not path.exists(images_folder_path):
            os.mkdir(images_folder_path)

        images_filenames = [images_folder_path + str(image_index) + ".png" for image_index in range(len(smiles_list))]

        # Debug
        #keypoints_list = self._draw_process(images_filenames, smiles_list) 
        
        images_filenames_split = np.array_split(images_filenames, self.config["num_processes_mp"])
        smiles_list_split = np.array_split(smiles_list, self.config["num_processes_mp"])

        keypoints_list = []
        args = [[images_filenames_split[process_index], smiles_list_split[process_index]] for process_index in range(self.config["num_processes_mp"])]
        pool = multiprocessing.Pool(self.config["num_processes_mp"])
        keypoints_processes = pool.map(self._draw_process_star, args)
        pool.close()
        pool.join()
        for index in range(self.config["num_processes_mp"]):
            keypoints_list.extend(keypoints_processes[index])
        
        molfiles_filenames = [image_filename[:-4] + ".mol" for image_filename in images_filenames]
        self.save_images_filenames_keypoints(keypoints_list, images_filenames, molfiles_filenames, smiles_list, self.synthetic_save_path)

    def get_cede_dataset(self):
        cede_path = os.path.dirname(__file__) + "/../../data/CEDe_10k/"

        labeled_data = LabeledData(
            root_path = cede_path, 
            labeled_data_path = "CEDe_synthetic_data_10k.json", 
            gt_data_path = ".",
            dataset_name = "UOB"
        )

        images_filenames = []
        annotations = []
        keypoints_list = []
        i = 0
        for cede_image_name in tqdm(labeled_data.images.index):
            if i == self.config["nb_sample"]:
                break
            
            annotation = labeled_data.annotations[labeled_data.annotations.image_id == labeled_data.images.loc[cede_image_name].id]
            # Filter stereo-chemistry 
            #if any([(isinstance(c, str) and c != "CHI_UNSPECIFIED") for c in annotation["atom_chiral"].values]) or \
            #    any([(isinstance(s, str) and s != "STEREONONE") for s in annotation["bond_stereo"].values]):
            #    continue 

            images_filenames.append(cede_path + cede_image_name)
            annotations.append(annotation)
            
            scaling = self.config["image_size"][1]/768
            keypoints = []
            for _, a in annotation.iterrows():
                if isinstance(a["element"], str) or (not math.isnan(a["element"])):
                    center_x = int((a["bbox"][0] + (a["bbox"][2]/2))*scaling)
                    center_y = int((a["bbox"][1] + (a["bbox"][3]/2))*scaling)
                    keypoints.append([center_x, center_y])
            keypoints_list.append(keypoints)
            i += 1
        
        dataset = pd.DataFrame({
            "image_filename": images_filenames, 
            "annotation": annotations,
            "keypoints": keypoints_list
        })
        return dataset 

    def setup(self, stage=None, hf_dataset=True):
        """
        setup() is automatically called on each GPU process.
        A distributedsampler under the hood will make the dataloaders draw samples from different part of the same dataset. 
        """
        print(f"Setting up data module, stage: {stage}")
        if self.config["on_fly"]:
            self.train_dataset = self.dataset_class([None]*1000, config=self.config)
            self.val_dataset = self.dataset_class([None]*1000, config=self.config)
            return

        if (stage == TrainerFn.FITTING) and (self.config["training_dataset"] == "cede-synthetic"):
            print("Setup CEDe dataset")
            dataset = self.get_cede_dataset()
            
            # Split
            train_ratio = 0.8
            val_ratio = 0.1
            test_ratio = 0.1
            train_dataset, test_dataset = train_test_split(dataset, test_size = 1 - train_ratio)
            val_dataset, test_dataset = train_test_split(test_dataset, test_size = test_ratio/(test_ratio + val_ratio)) 

            self.train_dataset = self.dataset_class(train_dataset, config=self.config, train=True)
            self.val_dataset = self.dataset_class(val_dataset, config=self.config, train=False)
            return 
        
        # Real Data
        if (stage == TrainerFn.FITTING) and (self.mode == "fine-tuning"):
            print("Setup real dataset")
            self.setup_keypoints_benchmarks(stage = "fine-tuning")
        
        predict = False
        if (stage != TrainerFn.FITTING):
            print("Disable augmentations for testing")
            # Testing without augmentations
            predict = True
                
        # Synthetic Data
        print("Setup synthetic dataset")
        if not hf_dataset:
            with open(os.path.dirname(__file__) + "/../../data/synthetic_images_molecules_keypoints/keypoints_images_filenames_" + self.config["experiment_name"] + ".json", 'r') as json_file:
                coco_json  = json.load(json_file)
            
            dataset = pd.DataFrame({
                "smiles": [image["smiles"] for index, image in enumerate(coco_json["images"]) if index < self.config["nb_sample"]], 
                "image_filename": [image["image_filename"] for index, image in enumerate(coco_json["images"]) if index < self.config["nb_sample"]], 
                "molfile_filename": [image["molfile_filename"] for index, image in enumerate(coco_json["images"]) if index < self.config["nb_sample"]],
                "keypoints": [annotation["keypoints"] for index, annotation in enumerate(coco_json["annotations"]) if index < self.config["nb_sample"]]
            })

            # Temporary fix
            dataset["image_filename"] = [p.replace("joker", "molgrapher") for p in dataset["image_filename"]]
            dataset["molfile_filename"] = [p.replace("joker", "molgrapher") for p in dataset["molfile_filename"]]

            # Filter
            if self.clean_only:
                print(f"Synthetic dataset length before filtering: {len(dataset)}")
                remove_indices = []
                for index, molfile_filename in enumerate(dataset["molfile_filename"]):
                    # Note: The index in the dataset df is range(0, len(df)) 
                    molfile_annotator = LabelMolFile(
                        molfile_filename,
                        reduce_abbreviations = True
                    ) 
                    molecule = molfile_annotator.rdk_mol
                    if any([((atom.GetSymbol() == "R") or (atom.GetSymbol() == "*") or atom.HasProp("_displayLabel")) for atom in molecule.GetAtoms()]):
                        remove_indices.append(index)
                dataset = dataset.drop(remove_indices)
                print(f"Synthetic dataset length after filtering: {len(dataset)}")

            # Split
            train_ratio = 0.8
            val_ratio = 0.1
            test_ratio = 0.1
            train_dataset, test_dataset = train_test_split(dataset, test_size = 1 - train_ratio)
            val_dataset, test_dataset = train_test_split(test_dataset, test_size = test_ratio/(test_ratio + val_ratio)) 

            # Save train/val/test splits
            train_filenames = pd.DataFrame([p.split("/")[-1] for p in train_dataset["image_filename"]])
            train_filenames.to_csv(os.path.dirname(__file__) + "/../../data/synthetic_images_molecules_keypoints/train_images_filenames_" + self.config["experiment_name"] + ".csv")
            val_filenames = pd.DataFrame([p.split("/")[-1] for p in val_dataset["image_filename"]])
            val_filenames.to_csv(os.path.dirname(__file__) + "/../../data/synthetic_images_molecules_keypoints/val_images_filenames_" + self.config["experiment_name"] + ".csv")
            test_filenames = pd.DataFrame([p.split("/")[-1] for p in test_dataset["image_filename"]])
            test_filenames.to_csv(os.path.dirname(__file__) + "/../../data/synthetic_images_molecules_keypoints/test_images_filenames_" + self.config["experiment_name"] + ".csv")

            self.train_dataset = self.dataset_class(train_dataset, config=self.config, train=True)
            self.val_dataset = self.dataset_class(val_dataset, config=self.config, train=False, predict=predict, evaluate=self.dataset_evaluate)
        
        if hf_dataset:
            from datasets import load_dataset
            dataset = load_dataset("ds4sd/molgrapher-synthetic-300k")

            train_dataset = dataset["train"].to_pandas()
            val_dataset = dataset["validation"].to_pandas()
            train_dataset["image_filename"] = [str(sample["id"]) + ".png" for _, sample in train_dataset.iterrows()]
            train_dataset["molfile_filename"] = [str(sample["id"]) + ".mol" for _, sample in train_dataset.iterrows()]
            val_dataset["image_filename"] = [str(sample["id"]) + ".png" for _, sample in val_dataset.iterrows()]
            val_dataset["molfile_filename"] = [str(sample["id"]) + ".mol" for _, sample in val_dataset.iterrows()]
            
            self.train_dataset = self.dataset_class(train_dataset, config=self.config, train=True, hf_dataset=hf_dataset)
            self.val_dataset = self.dataset_class(val_dataset, config=self.config, train=False, predict=predict, evaluate=self.dataset_evaluate, hf_dataset=hf_dataset)

    def save_images_filenames_keypoints(self, keypoints_list, images_filenames, molfiles_filenames, smiles_list, save_path):
        # COCO dataset
        coco_json = {
            "info": {
                    "year": "", 
                    "version": 0, 
                    "description": "", 
                    "contributor": "", 
                    "url": "", 
                    "date_created": ""
            },
            "licenses": [{"id": 0, "name": "", "url": ""}],
            "images": [],
            "annotations": []
        }
        
        # Save images
        for image_index, (keypoints, image_filename, molfile_filename, smiles) in tqdm(enumerate(zip(keypoints_list, images_filenames, molfiles_filenames, smiles_list)), total=len(images_filenames)):
            try:
                image = Image.open(image_filename) 
            except:
                continue
            # Clean dataset
            if (image is not None) and (keypoints is not None):
                coco_image = {
                    "width": int(image.size[0]), 
                    "height": int(image.size[1]), 
                    "image_filename": image_filename,
                    "molfile_filename": molfile_filename,
                    "smiles": smiles,
                    "id": image_index, 
                    "license": 0, 
                    "date_captured": "2022-10-05"
                }
                coco_json["images"].append(coco_image)

                flat_keypoints = []
                for keypoint in keypoints:
                    flat_keypoints.append(keypoint[0])
                    flat_keypoints.append(keypoint[1])
                    visibility = 2
                    flat_keypoints.append(visibility)

                annotation = {
                    "image_id": image_index, 
                    "keypoints": flat_keypoints,
                    "iscrowd": 0                     
                }
                coco_json["annotations"].append(annotation)
        
        with open(save_path, 'w') as outfile:
            json.dump(coco_json, outfile)

    def precompute_keypoints_benchmarks(self, dataset_max_index, dataset):
        self.test_window_size = [20, 20]

        current_dataset_index = 0
        
        images_folder_path = os.path.dirname(__file__) + "/../../data/benchmarks/" + dataset + "/images/"
        molfiles_folder_path = os.path.dirname(__file__) + "/../../data/benchmarks/" + dataset + "/molfiles/"
        synthetic_images_folder_path = os.path.dirname(__file__) + "/../../data/benchmarks/" + dataset + "/synthetic_images/"
        if not os.path.exists(synthetic_images_folder_path):
            os.makedirs(synthetic_images_folder_path)

        save_path = os.path.dirname(__file__) + "/../../data/benchmarks/" + dataset + "/keypoints_images_filenames_" + \
                    self.config["experiment_name"] + "_" + str(dataset_max_index) + "_clean_" + str(int(self.clean_only)) + ".json"

        molfiles_filenames = []
        images_filenames = []

        for molfile_filename in glob.glob(molfiles_folder_path + "*"):
            if current_dataset_index >= dataset_max_index:
                break
        
            # Filter input molecules
            image_path = images_folder_path + molfile_filename[:-4].split("/")[-1] + ".png" 
            
            # Special case for JPO evaluation
            if dataset == "jpo_c":
                molfiles_filenames.append(molfile_filename) 
                images_filenames.append(image_path)
                current_dataset_index += 1
                continue 

            if dataset == "jpo":
                print("JPO filtering is not supported")
                molfiles_filenames.append(molfile_filename) 
                images_filenames.append(image_path)
                current_dataset_index += 1
                continue  

            molfile_annotator = LabelMolFile(molfile_filename)
            are_markush, are_molecules, are_abbreviated, nb_structures = molfile_annotator.are_markush(), \
                                                                            molfile_annotator.are_molecules(), \
                                                                            molfile_annotator.are_abbreviated(), \
                                                                            molfile_annotator.how_many_structures()

            if (not any(are_markush) and not any(are_abbreviated) and not any(are_molecules)):
                print("The MolFile annotation is not supported")
                continue
            
            if not path.exists(image_path):
                print(f"No image corresponding to the molfile {molfile_filename} was found")
                continue

            if self.clean_only:
                if (are_markush == [False]) and (are_abbreviated == [False]) and (nb_structures == 1):
                    molfiles_filenames.append(molfile_filename) 
                    images_filenames.append(image_path)

            else:
                if (are_markush == [False]) and (nb_structures == 1):
                    molfiles_filenames.append(molfile_filename) 
                    images_filenames.append(image_path)

            current_dataset_index += 1

        keypoints_list = []
        remove_indices = []
        for dataset_index, (molfile_filename, image_path) in tqdm(enumerate(zip(molfiles_filenames, images_filenames)), total=len(molfiles_filenames)):
            # Compute keypoints
            synthetic_image_path = synthetic_images_folder_path + molfile_filename[:-4].split("/")[-1] + ".png"

            molfile_annotator = LabelMolFile(
                molfile_filename,
                reduce_abbreviations = True
            )
            molecule = molfile_annotator.rdk_mol 
            try:
                molecule.UpdatePropertyCache()
            except:
                print("Keypoints generation error: The molecule property cache can not be updated")
                remove_indices.append(dataset_index)
                continue
            
            try:
                # Manual Cleaning
                if any((name in synthetic_image_path) for name in ["US20220281806A1-20220908-C00012", "US20220281806A1-20220908-C00019"]):
                    remove_indices.append(dataset_index)
                    continue
                #print(synthetic_image_path)
                _, keypoints = draw_molecule_keypoints_rdkit(
                    smiles = Chem.MolToSmiles(molecule), 
                    molecule = molecule, 
                    path = synthetic_image_path, 
                    augmentations = False,
                    fake_molecule = False,
                    save_molecule = False
                )
            except:
                print("Keypoints generation error: The molecule can not be drawn using RDKit")
                remove_indices.append(dataset_index)
                continue

            if keypoints is None:
                print("Keypoints generation error: The molecule can not be drawn using RDKit")
                remove_indices.append(dataset_index)
                continue

            keypoints_list.append(keypoints)

        images_filenames = [image_filename for i, image_filename in enumerate(images_filenames) if i not in remove_indices]
        molfiles_filenames = [molfile_filename for i, molfile_filename in enumerate(molfiles_filenames) if i not in remove_indices]

        keypoints_list_transformed = []
        remove_indices = []
        for dataset_index, (molfile_filename, image_filename) in tqdm(enumerate(zip(molfiles_filenames, images_filenames)), total=len(molfiles_filenames)):
            
            synthetic_image_filename = synthetic_images_folder_path + molfile_filename[:-4].split("/")[-1] + ".png"
            keypoints = keypoints_list[dataset_index]

            # Transform ground truth image
            image_gt = transform_png_image(image_filename)              
            image_gt = resize_image(
                image_gt, 
                image_size = (self.config["image_size"][1], 
                self.config["image_size"][2]), 
                border_size = 30 #TODO
            )
            image_gt = np.array(image_gt)
            image_gt = np.stack((image_gt,)*3, axis=-1)

            image_rdkit = cv2.imread(synthetic_image_filename) 
            transformation, margin = get_affine_transformation(image_rdkit, image_gt)

            if transformation is not None:
                # Transform the keypoints
                keypoints_transformed = self.transform_keypoints(keypoints, transformation, margin)
            else:
                print(f"Tranformation discarded {dataset_index}")
                remove_indices.append(dataset_index)
                continue

            empty_keypoint = False
            for keypoint in keypoints_transformed:
                image_crop = functional.crop(
                    torch.from_numpy(image_gt).permute(2, 0, 1), 
                    top = keypoint[1] - (self.test_window_size[1]//2),
                    left = keypoint[0] - (self.test_window_size[0]//2),
                    height = self.test_window_size[1],
                    width = self.test_window_size[0]
                )
                if (image_crop != 255.).sum() < 1:
                    empty_keypoint = True
                    break

            if empty_keypoint:
                print(f"Tranformation discarded {dataset_index}. Some aligned keypoints match empty image regions.")
                remove_indices.append(dataset_index)
                continue

            keypoints_list_transformed.append(keypoints_transformed)

        print(f"Proportion of removed samples in {dataset}: {round(len(remove_indices)/len(images_filenames), 4)}")
        images_filenames = [image_filename for i, image_filename in enumerate(images_filenames) if i not in remove_indices]
        molfiles_filenames = [molfile_filename for i, molfile_filename in enumerate(molfiles_filenames) if i not in remove_indices]
        smiles_list = ["" for molfile_filename in molfiles_filenames]
        self.save_images_filenames_keypoints(keypoints_list_transformed, images_filenames, molfiles_filenames, smiles_list, save_path)

    def setup_keypoints_benchmarks(self, stage="test"):
        # Fine-tuning
        if stage == "fine-tuning":
            dataset_max_index = self.config["nb_sample_fine-tuning"]
            save_path = os.path.dirname(__file__) + "/../../data/benchmarks/" + self.config["training_dataset_fine-tuning"] + "/keypoints_images_filenames_" + \
                        self.config["experiment_name"] + "_" + str(dataset_max_index) + "_clean_" + str(int(self.clean_only)) + ".json"
            
            # Create dataset
            if (not os.path.exists(save_path)) or self.force_precompute:
                print(f"Precompute keypoints for fine-tuning training dataset: {self.config['training_dataset_fine-tuning']}")
                self.precompute_keypoints_benchmarks(dataset_max_index, self.config["training_dataset_fine-tuning"])
            
            # Read dataset
            with open(save_path, 'r') as json_file:
                coco_json  = json.load(json_file)
            dataset_keypoints = [annotation["keypoints"] for index, annotation in enumerate(coco_json["annotations"]) if index < dataset_max_index]
            images_filenames = [image["image_filename"] for index, image in enumerate(coco_json["images"]) if index < dataset_max_index]
            molfiles_filenames = [image["molfile_filename"] for index, image in enumerate(coco_json["images"]) if index < dataset_max_index]
            smiles_list = [None for index, image in enumerate(coco_json["images"]) if index < dataset_max_index] 
            dataset = pd.DataFrame({"smiles": smiles_list, "image_filename": images_filenames, "molfile_filename": molfiles_filenames, "keypoints": dataset_keypoints})
            
            # Temporary fix
            images_filenames = [p.replace("joker", "molgrapher") for p in images_filenames]
            molfiles_filenames = [p.replace("joker", "molgrapher") for p in molfiles_filenames]

            # Split training/validation/test
            train_ratio = 0.8
            val_ratio = 0.1
            test_ratio = 0.1
            train_dataset, test_dataset = train_test_split(dataset, test_size = 1 - train_ratio)
            val_dataset, test_dataset = train_test_split(test_dataset, test_size = test_ratio/(test_ratio + val_ratio)) 
            self.train_dataset_real = self.dataset_class(
                train_dataset, 
                config = self.config, 
                train = True
            )
            self.val_dataset_real = self.dataset_class(
                val_dataset, 
                config = self.config, 
                train = False, 
                predict = True, # Validation without augmentations
                evaluate = self.dataset_evaluate
            )

        # Testing
        self.benchmarks_datasets = [] 
        for benchmark, dataset_max_index in zip(self.config["benchmarks"], self.config["nb_sample_benchmarks"]):
            save_path = os.path.dirname(__file__) + "/../../data/benchmarks/" + benchmark + "/keypoints_images_filenames_" + \
                        self.config["experiment_name"] + "_" + str(dataset_max_index) + "_clean_" + str(int(self.clean_only)) + ".json"
            
            # Create dataset
            if (not os.path.exists(save_path)) or self.force_precompute:
                print(f"Precompute keypoints for benchmark: {benchmark}")
                self.precompute_keypoints_benchmarks(dataset_max_index, benchmark)
            
            # Read dataset
            with open(save_path, 'r') as json_file:
                coco_json  = json.load(json_file)
            dataset_keypoints = [annotation["keypoints"] for index, annotation in enumerate(coco_json["annotations"]) if index < dataset_max_index]
            images_filenames = [image["image_filename"] for index, image in enumerate(coco_json["images"]) if index < dataset_max_index]
            molfiles_filenames = [image["molfile_filename"] for index, image in enumerate(coco_json["images"]) if index < dataset_max_index]
            smiles_list = [None for index, image in enumerate(coco_json["images"]) if index < dataset_max_index] 
            dataset = pd.DataFrame({"smiles": smiles_list, "image_filename": images_filenames, "molfile_filename": molfiles_filenames, "keypoints": dataset_keypoints})
            
            # Temporary fix
            images_filenames = [p.replace("joker", "molgrapher") for p in images_filenames]
            molfiles_filenames = [p.replace("joker", "molgrapher") for p in molfiles_filenames]
            
            self.benchmarks_datasets.append(self.dataset_class(dataset, config=self.config, train=False, predict=True, evaluate=self.dataset_evaluate))

    def setup_molfiles_benchmarks_static(self, benchmark, nb_sample_benchmark):
        min_index = 2500
        molfiles_filenames = []
        images_filenames = []
        images_folder_path = os.path.dirname(__file__) + "/../../data/benchmarks/" + benchmark + "/images/"
        molfiles_folder_path = os.path.dirname(__file__) + "/../../data/benchmarks/" + benchmark + "/molfiles/"
        for molfile_filename in glob.glob(molfiles_folder_path + "*"):
            image_path = images_folder_path + molfile_filename[:-4].split("/")[-1] + ".png"
            if os.path.exists(image_path):
                # Debugging
                #if "US20220251141A1-20220811-C00016" not in image_path:
                #    continue
                molfiles_filenames.append(molfile_filename) 
                images_filenames.append(image_path)
            else:
                print(f"Warning: {image_path} doesn't match {molfile_filename}")
            
        if len(images_filenames) == 0:
            print(f"Error: the benchmark {benchmark} is empty")
        images_filenames = [image_filename for index, image_filename in enumerate(images_filenames) if (index < nb_sample_benchmark) and (index > min_index)]
        molfiles_filenames = [molfile_filename for index, molfile_filename in enumerate(molfiles_filenames) if (index < nb_sample_benchmark) and (index > min_index)]
        smiles_list = [None for index, molfile_filename in enumerate(molfiles_filenames)] 
        dataset = pd.DataFrame({"smiles": smiles_list, "image_filename": images_filenames, "molfile_filename": molfiles_filenames})

        # Filter molecule images with abbreviations
        if self.clean_only:
            print(f"Benchmark dataset length before filtering: {len(dataset)}")
            remove_indices = []
            for index, molfile_filename in enumerate(dataset["molfile_filename"]):
                # Note: The index in the dataset df is range(0, len(df)) 
                molfile_annotator = LabelMolFile(
                    molfile_filename,
                    reduce_abbreviations = True
                ) 
                molecule = molfile_annotator.rdk_mol
                if any([((atom.GetSymbol() == "R") or (atom.GetSymbol() == "*") or atom.HasProp("_displayLabel")) for atom in molecule.GetAtoms()]):
                    remove_indices.append(index)
            dataset = dataset.drop(remove_indices)
            print(f"Benchmark dataset length after filtering: {len(dataset)}")

        self.benchmarks_datasets = [
            self.dataset_class(
                dataset, 
                config=self.config, 
                train=False, 
                predict=self.dataset_predict, 
                return_images_filenames=self.return_images_filenames, 
                evaluate=self.dataset_evaluate,
                taa_step=self.taa_step
            )]
        print("Dataset: ", self.benchmarks_datasets[0].dataset)

    def setup_images_benchmarks(self, max_index=None):
        if self.images_folder_path != None:
            images_filenames = [
                image_filename 
                    for image_filename in glob.glob(self.images_folder_path + "/*")
                    if ((".png" in image_filename) or (".TIF" in image_filename))
            ]
            
            if len(images_filenames) == 0:
                print(f"Image directory not found: {self.images_folder_path}")

        elif self.images_paths != None:
            images_filenames = self.images_paths

            if len(images_filenames) == 0:
                print(f"Empty image list: {self.images_paths}")

        # Skip pre-processed
        images_filenames_clean = []
        for image_filename in images_filenames:
            if "preprocessed" in image_filename:
                print(f"Skipping image file: {image_filename}")
            else:
                images_filenames_clean.append(image_filename)

        dataset = pd.DataFrame(images_filenames_clean, columns=["image_filename"])   
        self.benchmarks_datasets = [self.dataset_class(dataset, config=self.config, force_cpu=self.force_cpu)]

    def preprocess(self):
        from molgrapher.utils.utils_dataset import CaptionRemover
        for benchmark_dataset in self.benchmarks_datasets:
            self.caption_remover = CaptionRemover(self.config, force_cpu=self.force_cpu, remove_captions=self.remove_captions)
            preprocessed_images = self.caption_remover.preprocess_images(benchmark_dataset.dataset["image_filename"])
            benchmark_dataset.preprocessed = True

            # Save
            preprocessed_images_filenames = []
            for preprocessed_image, image_filename in zip(preprocessed_images, benchmark_dataset.dataset["image_filename"]):
                preprocessed_image_filename = image_filename[:-4] + "_preprocessed.png"
                preprocessed_image.save(preprocessed_image_filename)
                preprocessed_images_filenames.append(preprocessed_image_filename)
            benchmark_dataset.dataset["image_filename"] = preprocessed_images_filenames

    def transform_keypoints(self, keypoints, transformation, margin):
        keypoints = [[keypoint[0] + margin, keypoint[1] + margin] for keypoint in keypoints]
        transformed_keypoints = cv2.transform(np.array([keypoints]), transformation)[0]
        transformed_keypoints = [[int(keypoint[0] - margin), int(keypoint[1] - margin)] for keypoint in transformed_keypoints]
        return transformed_keypoints

    def train_dataloader(self):
        """
            Lightning convert this dataloader to a distributed dataloader.
        """
        print("Setting up training dataloader")
        if self.mode == "fine-tuning":
            print(f"Real train set length: {len(self.train_dataset_real)}")
            return self.get_dataloader(self.train_dataset_real)
        else:
            print(f"Synthetic train set length: {len(self.train_dataset)}")
            return self.get_dataloader(self.train_dataset)
        
    def val_dataloader(self):
        """
        Ligthning create python processes for each training and validation worker. 
        During training, the validation workers remain idle. 
        """
        if self.mode == "fine-tuning":
            print("Setting up multiple validation dataloaders (fine-tuning)")
            print(f"Synthetic validation set length: {len(self.val_dataset)} (1/10 of total dataset size)")
            print(f"Real validation set length: {len(self.val_dataset_real)} (1/10 of total dataset size)")
            print(f"Real benchmarks sets lengths (on-the-fly validation): {[len(benchmark_dataset) for benchmark_dataset in self.benchmarks_datasets]}")
            dataloaders_list = []
            
            # Synthetic validation
            dataloaders_list.append(self.get_dataloader(self.val_dataset))

            # Real data validation
            dataloaders_list.append(self.get_dataloader(self.val_dataset_real))

            # On-the-fly benchmark
            for benchmark_dataset in self.benchmarks_datasets:
                dataloaders_list.append(self.get_dataloader(benchmark_dataset))

            print(f"Number of validation sets: {len(dataloaders_list)}")
            return dataloaders_list
        else:
            print("Setting up validation dataloader")
            return self.get_dataloader(self.val_dataset)

    def predict_dataloader(self):
        """
        Note: prefetch_factor is given per worker.
        """
        dataloaders_list = []
        print("Setting up predict dataloader")
        print(f"Real benchmarks sets lengths (on-the-fly validation): {[len(benchmark_dataset) for benchmark_dataset in self.benchmarks_datasets]}")
        for benchmark_dataset in self.benchmarks_datasets:
            dataloaders_list.append(self.get_dataloader(benchmark_dataset))
        print(f"Number of predict sets: {len(dataloaders_list)}")
        return dataloaders_list

    def get_dataloader(self, dataset):
        if hasattr(dataset, 'collate_fn') and (dataset.collate_fn != None):
            return DataLoader(
                dataset, 
                batch_size = self.config["batch_size"], 
                num_workers = self.config["nb_workers"], 
                shuffle = False, 
                prefetch_factor = 2,
                pin_memory = True,  
                persistent_workers = True,
                drop_last = False,
                collate_fn = dataset.collate_fn
            )
        else:
            return torch_geometric.loader.DataLoader(
                dataset, 
                batch_size = self.config["batch_size"], 
                num_workers = self.config["nb_workers"], 
                shuffle = False, 
                prefetch_factor = 2,
                pin_memory = True,  
                persistent_workers = True,
                drop_last = False,
            )
            
