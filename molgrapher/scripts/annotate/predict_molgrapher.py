from rdkit import Chem
from rdkit.Chem import rdmolfiles
import json
import os
import argparse
import pandas as pd
import ast
from pprint import pprint 
from time import time 
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from tqdm import tqdm 
import cv2
import shutil 
import torch 
from more_itertools import chunked
from PIL import Image 

from molgrapher.models.graph_recognizer import GraphRecognizer
from molgrapher.models.abbreviation_detector import AbbreviationDetectorCPU, AbbreviationDetectorGPU, SpellingCorrector
from molgrapher.datasets.dataset_image import ImageDataset
from molgrapher.data_modules.data_module import DataModule
from molgrapher.utils.utils_dataset import get_bonds_sizes
from mol_depict.utils.utils_generation import get_abbreviations_smiles_mapping
from mol_depict.utils.utils_drawing import draw_molecule_rdkit
from molgrapher.utils.utils_logging import count_model_parameters


os.environ["OMP_NUM_THREADS"] = "1" 
cv2.setNumThreads(0)
torch.set_float32_matmul_precision("medium")

def proceed_batch(args, input_images_paths):
    # Configuration
    model_name = "wedge_4"
    clean = True
    # Action
    predict = True
    visualize = False
    visualize_rdkit = False
    # DataLoader
    preprocess = True
    # Save
    clean = True
    visualize_output_folder_path = os.path.dirname(__file__) + "/../../../data/visualization/predictions/PatCID_1K/"
    visualize_rdkit_output_folder_path = os.path.dirname(__file__) + "/../../../data/predictions/rdkit/"
    
    if visualize:
        if clean and (os.path.exists(visualize_output_folder_path)):
            shutil.rmtree(visualize_output_folder_path)
        if not os.path.exists(visualize_output_folder_path):
            os.makedirs(visualize_output_folder_path)
    if predict and (args.save_mol_folder != ""):
        if clean and (os.path.exists(args.save_mol_folder)):
            shutil.rmtree(args.save_mol_folder)
        if not os.path.exists(args.save_mol_folder):
            os.makedirs(args.save_mol_folder)
            
    # Read config file
    with open(args.config_dataset_graph_path) as file:
        config_dataset_graph = json.load(file)
    with open(args.config_training_graph_path) as file:
        config_training_graph = json.load(file)
    with open(args.config_dataset_keypoint_path) as file:
        config_dataset_keypoint = json.load(file)
    with open(args.config_training_keypoint_path) as file:
        config_training_keypoint = json.load(file)
    # Update config 
    config_dataset_graph["num_processes_mp"] = args.num_processes_mp
    config_dataset_graph["num_threads_pytorch"] = args.num_threads_pytorch
    config_dataset_keypoint["num_processes_mp"] = args.num_processes_mp
    config_dataset_keypoint["num_threads_pytorch"] = args.num_threads_pytorch

    # Read dataset images
    data_module = DataModule(
        config_dataset_graph, 
        dataset_class = ImageDataset,
        images_folder_path = None,
        images_paths = input_images_paths,
        force_cpu = args.force_cpu,
        remove_captions = args.remove_captions
    )
    data_module.setup_images_benchmarks()
    if preprocess:
        print(f"Starting Caption Removal Preprocessing")
        ref_t = time()
        data_module.preprocess()
        print(f"Caption Removal Preprocessing completed in {round(time() - ref_t, 2)}")

    # Read model
    model = GraphRecognizer(
        config_dataset_keypoint, 
        config_training_keypoint, 
        config_dataset_graph, 
        config_training_graph
    )

    print(f"Keypoint detector number parameters: {round(count_model_parameters(model.keypoint_detector)/10**6, 4)} M")
    print(f"Node classifier number parameters: {round(count_model_parameters(model.graph_classifier)/10**6, 4)} M")

    # Set up trainer
    if args.force_cpu:
        trainer = pl.Trainer(
            accelerator = "cpu",
            precision = config_training_graph["precision"],
            logger = False
        )
    else:
        trainer = pl.Trainer(
            accelerator = config_training_graph["accelerator"],
            devices = config_training_graph["devices"],
            precision = config_training_graph["precision"],
            logger = False
        )

    # Get predictions
    torch.set_num_threads(config_dataset_graph["num_threads_pytorch"])
    print(f"Starting Keypoint Detection + Node Classification")
    ref_t = time()
    predictions_out = trainer.predict(model, dataloaders=data_module.predict_dataloader())
    print(f"Keypoint Detection + Node Classification completed in {round(time() - ref_t, 2)}")
    
    images_filenames = []
    images = []
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
            images.append(_elem)
    
    scaling_factor = config_dataset_keypoint["image_size"][1]//config_dataset_keypoint["mask_size"][1]
    
    # Compute bond size
    bonds_sizes = get_bonds_sizes(predictions["keypoints"], scaling_factor)
            
    # Recognize abbreviations
    print(f"Starting Abbreviation Recognition")
    ref_t = time()
    if args.force_cpu or config_training_graph["accelerator"] == "cpu":
        abbreviation_detector = AbbreviationDetectorCPU(config_dataset_graph, force_cpu = args.force_cpu)
    else:
        abbreviation_detector = AbbreviationDetectorGPU(config_dataset_graph, force_cpu = args.force_cpu)
    abbreviations_list = abbreviation_detector.mp_run(images_filenames, predictions["graphs"], bonds_sizes)
    print(f"Abbreviation Recognition completed in {round(time() - ref_t, 2)}")

    # Create RDKit graph
    print("Starting Graph creation")
    ref_t = time()
    with open(os.path.dirname(__file__) + "/../../../data/ocr_mapping/ocr_atoms_classes_mapping.json") as file:
        ocr_atoms_classes_mapping = json.load(file)
    abbreviations_smiles_mapping = get_abbreviations_smiles_mapping()
    predicted_molecules = []
    for abbreviations, graph in zip(abbreviations_list, predictions["graphs"]):
        predicted_molecule = graph.to_rdkit(
            abbreviations, 
            abbreviations_smiles_mapping, 
            ocr_atoms_classes_mapping, 
            SpellingCorrector(abbreviations_smiles_mapping),
            assign_stereo = args.assign_stereo,
            align_rdkit_output = args.align_rdkit_output
        ) 
        predicted_molecules.append(predicted_molecule)
    print(f"Graph creation completed in {round(time() - ref_t, 2)}")
    predictions["molecules"] = predicted_molecules

    # Convert to SMILES and set confidence
    predictions["smiles"] = []
    for i, (predicted_molecule, image_filename) in enumerate(zip(predictions["molecules"], images_filenames)):
        if args.save_mol_folder != "":
            molecule_path = args.save_mol_folder + image_filename.split("/")[-1][:-4].replace("_preprocessed", "") + ".mol"
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
    for predicted_smiles, confidence, image_filename, abbreviations in zip(predictions["smiles"], predictions["confidences"], input_images_paths, abbreviations_list):
        #annotation_filename = image_filename.split("/")[1].split(".")[0] + ".jsonl"
        annotation_filename = args.save_mol_folder + "smiles.jsonl"
        with open(annotation_filename, "a") as f:
            if predicted_smiles is not None:
                if abbreviations != []:
                    abbreviations_texts = [abbreviation["text"] for abbreviation in abbreviations]
                else:
                    abbreviations_texts = []

                annotation = {
                    "smi": predicted_smiles,
                    "abbreviations": abbreviations_texts,
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
            
            json.dump(annotation, f)
            f.write('\n')

    print("Annotation:")
    print(pd.read_json(path_or_buf = annotation_filename, lines = True))
    

    # Visualize predictions
    if visualize:
        for image_filename, image, graph, keypoints, molecule in tqdm(zip(images_filenames, images, predictions["graphs"], predictions["keypoints"], predictions["molecules"]), total=len(images_filenames)):
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

            plt.savefig(f"{visualize_output_folder_path}/{image_filename.split('/')[-1]}")
            plt.close()

    if visualize_rdkit:
        for image_filename in input_images_paths:
            molecule_path = args.save_mol_folder + image_filename.split("/")[-1][:-4].replace("_preprocessed", "") + ".mol"
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
                plt.savefig(f"{visualize_rdkit_output_folder_path}/{image_filename.split('/')[-1]}")
            plt.close()
            
        
    
def main():
    starting_time = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-images-paths', type = str)
    parser.add_argument('--force-cpu', action = argparse.BooleanOptionalAction, default = True, required = False)
    parser.add_argument('--num-threads-pytorch', type = int, default = 10)
    parser.add_argument('--num-processes-mp', type = int, default = 10)
    parser.add_argument('--chunk-size', type = int, default = 200)
    parser.add_argument('--assign-stereo', action = argparse.BooleanOptionalAction, default = True, required = False)
    parser.add_argument('--align-rdkit-output', type = bool, default = False)
    parser.add_argument('--remove-captions', action = argparse.BooleanOptionalAction, default = True, required = False)
    parser.add_argument('--save-mol-folder', type = str, default = "")
    parser.add_argument('--config-dataset-graph-path', type = str, default = os.path.dirname(__file__) + "/../../../data/config_dataset_graph_2.json")
    parser.add_argument('--config-training-graph-path', type = str, default = os.path.dirname(__file__) + "/../../../data/config_training_graph.json")
    parser.add_argument('--config-dataset_keypoint-path', type = str, default = os.path.dirname(__file__) + "/../../../data/config_dataset_keypoint.json")
    parser.add_argument('--config-training-keypoint-path', type = str, default = os.path.dirname(__file__) + "/../../../data/config_training_keypoint.json")
    args = parser.parse_args()

    print("Arguments:")
    pprint(vars(args))

    with open(args.input_images_paths, 'r') as f:
        _input_images_paths = []
        for line in f.readlines():
            _input_images_paths.append(ast.literal_eval(line.strip())["path"])
        print("Number of images to annotate: ", len(_input_images_paths))

    for _batch_images_paths in chunked(_input_images_paths, args.chunk_size):
        proceed_batch(args, _batch_images_paths)
    print(f"Annotation completed in: {round(time() - starting_time, 2)}")

if __name__ == "__main__":
    main()
