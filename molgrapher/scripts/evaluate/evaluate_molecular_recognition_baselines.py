#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdkit import Chem
import json
import os
from pprint import pprint
import pytorch_lightning as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image 
import shutil 
import pandas as pd 

try:
    from molgrapher.models.decimer import Decimer
except Exception as e: 
    print(f"Decimer not imported: {str(e)}")
try:
    from molgrapher.models.molscribe import Molscribe
except Exception as e: 
    print(f"MolScribe not imported: {str(e)}")
try:
    from molgrapher.models.img2mol import Img2Mol
except Exception as e: 
    print(f"Img2Mol not imported: {str(e)}")
from molgrapher.models.osra import OSRA
from molgrapher.models.molvec import MolVec
from molgrapher.models.imago import Imago
from molgrapher.data_modules.data_module import DataModule
from molgrapher.datasets.dataset_molfile import MolfileDataset
from molgrapher.utils.utils_evaluation import compute_molecule_prediction_quality, get_molecule_information
from mol_depict.utils.utils_drawing import draw_molecule_rdkit


def main():
    model_name = "img2mol"
    precompute = False
    visualize = False
    evaluate = True
    visualize_outliers = False
    clean = True
    save = True 

    # Read config file
    with open(os.path.dirname(__file__) + "/../../../data/config_dataset_graph.json") as file:
        config_dataset_graph = json.load(file)

    scores_logging = {}
    for benchmark, nb_sample_benchmark in zip(config_dataset_graph["benchmarks"], config_dataset_graph["nb_sample_benchmarks"]):
    
        if save:
            save_smiles_path = os.path.dirname(__file__) + f"/../../../data/predictions/{model_name}/{benchmark}_{nb_sample_benchmark}.csv"

        if visualize or visualize_outliers:
            output_folder_path = os.path.dirname(__file__) + f"/../../../data/visualization/predictions/{model_name}/{benchmark}/"
            if clean and os.path.exists(output_folder_path):
                shutil.rmtree(output_folder_path)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

        # Read  dataset
        data_module = DataModule(
            config_dataset_graph, 
            dataset_class = MolfileDataset,
            return_images_filenames = True,
            dataset_predict = False
        )
        if precompute:
            data_module.precompute_keypoints_benchmarks()
        data_module.setup_molfiles_benchmarks_static(benchmark, nb_sample_benchmark)

        # Read model
        if model_name == "osra":
            model = OSRA()
        if model_name == "molvec":
            model = MolVec()
        if model_name == "imago":
            model = Imago()
        if model_name == "img2mol":
            model = Img2Mol()
        if model_name == "decimer":
            model = Decimer()
        if model_name == "molscribe":
            model = Molscribe(device = "cuda:0")

        if model_name == "molscribe":
            # MolScribe does not work with Lightning out of the box
            loader = data_module.predict_dataloader()[0]
            images_filenames = [image_filename for batch in loader for image_filename in batch["images_filenames"]]
            molfiles_filenames = [molfile_filename for batch in loader for molfile_filename in batch["molfiles_filenames"]]
            
            predicted_smiles_list = model.predict(images_filenames)
            
        if model_name != "molscribe":
            # Set up trainer
            trainer = pl.Trainer(
                accelerator = "gpu" if ((model == "decimer")) else "cpu",
                logger = False, 
            )

            loader = data_module.predict_dataloader()[0]
            images_filenames = [image_filename for batch in loader for image_filename in batch["images_filenames"]]
            molfiles_filenames = [molfile_filename for batch in loader for molfile_filename in batch["molfiles_filenames"]]

            predictions = trainer.predict(model, dataloaders=loader)
            predictions = [predictions_batch for predictions_batch in predictions if predictions_batch != None]
            predicted_smiles_list = [prediction for prediction_batch in predictions for prediction in prediction_batch]

        if save: 
            results = pd.DataFrame({
                "image_filename": [image_filename.split("/")[-1] for image_filename in images_filenames],
                "smiles": predicted_smiles_list
            })
            results.to_csv(save_smiles_path)

        if visualize:
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            for image_filename, smiles in tqdm(zip(images_filenames, predicted_smiles_list), total=len(images_filenames)):
                figure, axis = plt.subplots(1, 2, figsize=(20, 10))
                axis[0].imshow(Image.open(image_filename))
            
                image_rdkit = draw_molecule_rdkit(smiles, augmentations=False)
                axis[1].imshow(image_rdkit.permute(1, 2, 0))
                plt.savefig(f"{output_folder_path}/{image_filename.split('/')[-1]}", dpi=500)
                plt.close()

        if evaluate:
            scores = {}
            molecular_information = {}
            for image_filename, predicted_smiles, molfile_filename in zip(images_filenames, predicted_smiles_list, molfiles_filenames):
                image_name = image_filename.split("/")[-1][:-4]
                gt_molecule = Chem.MolFromMolFile(molfile_filename)
                if not gt_molecule:
                    print(f"Invalid MolFile: {molfile_filename}")
                    continue

                molecular_information[image_name] = get_molecule_information(molfile_filename = molfile_filename)

                try:
                    scores[image_name] = compute_molecule_prediction_quality(
                        predicted_smiles,
                        Chem.MolToSmiles(gt_molecule), 
                        gt_molecule = gt_molecule
                    )
                except:
                    print(f"Evaluation error: {molfile_filename}")
                    continue

            score_correct = [scores[molecule_index]["correct"] for molecule_index in scores.keys()]
            score_correct = sum(score_correct)/len(scores)
            
            scores_logging[benchmark] = {
                "number_input_images": len(images_filenames),
                "number_processed_images": len(scores),
                "molecular_precision": round(score_correct, 4)
            }
            pprint(scores_logging)
    
            with open(os.path.dirname(__file__) + "/../../../data/scores/molecular_recognition/scores_" + model_name + "_" + benchmark + ".json", 'w') as outfile:
                json.dump(scores, outfile)

            with open(os.path.dirname(__file__) + "/../../../data/scores/molecular_recognition/information_" + model_name + "_" + benchmark + ".json", 'w') as outfile:
                json.dump(molecular_information, outfile)


        # Visualize outliers
        if visualize_outliers:
            if not os.path.exists(output_folder_path + "/outliers/"):
                os.makedirs(output_folder_path + "/outliers/")

            for image_filename, predicted_smiles in tqdm(zip(images_filenames, predicted_smiles_list), total=len(images_filenames)):
                image_name = image_filename.split('/')[-1][:-4]
                if image_name not in scores:
                    continue
                    
                score = scores[image_name]

                # Select molecules with wrong predictions
                if (score["correct"] == True):
                    continue

                if (predicted_smiles is None) or (Chem.MolFromSmiles(predicted_smiles) is None):
                    #print("Invalid predicted SMILES: ", predicted_smiles)
                    predicted_smiles = "C"

                #print("Incorrect predicted SMILES: ", predicted_smiles)
                figure, axis = plt.subplots(1, 2, figsize=(20, 10))
                axis[0].imshow(Image.open(image_filename))
                image_rdkit = draw_molecule_rdkit(predicted_smiles, augmentations=False)
                if image_rdkit is not None:
                    axis[1].imshow(image_rdkit.permute(1, 2, 0))
                plt.title(f"Tanimoto similarity: {score['tanimoto']}")
                plt.savefig(f"{output_folder_path}/outliers/{image_filename.split('/')[-1]}")
                plt.close()

if __name__ == "__main__":
    main()