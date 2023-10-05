#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json 
from collections import defaultdict
import pandas as pd
from tqdm import tqdm 
import pytorch_lightning as pl
import matplotlib.pyplot as plt 
from torchvision import transforms
import numpy as np 

# Modules
from molgrapher.utils.utils_dataset import get_bond_size
from molgrapher.utils.utils_evaluation import get_metrics_keypoint_detector
from molgrapher.data_modules.data_module import DataModule
from molgrapher.models.keypoint_detector import KeypointDetector
from molgrapher.datasets.dataset_keypoint import KeypointDataset
from molgrapher.datasets.dataset_molfile import MolfileDataset


def main():
    keypoint_detector_model_name = "exp-ad-201-run-k10-val_loss=0.0003-v2"
    precompute = False
    output_folder_path = os.path.dirname(__file__) + f"/../../../data/visualization/predictions/keypoint_detector/jpo_10/"
    visualize = True
    force_cpu = False
    visualize_outliers = False
    evaluate = True

    # Read config files
    with open(os.path.dirname(__file__) + "/../../../data/config_dataset_keypoint.json") as file:
        config_dataset = json.load(file)

    with open(os.path.dirname(__file__) + "/../../../data/config_training_keypoint.json") as file:
        config_training = json.load(file)

    for benchmark, nb_sample_benchmark in zip(config_dataset['benchmarks'], config_dataset["nb_sample_benchmarks"]):
        # Read dataset
        if evaluate:
            data_module = DataModule(
                config_dataset, 
                dataset_class = KeypointDataset, 
                mode = "train", 
                force_precompute = precompute
            )

            if benchmark == "molgrapher-synthetic":
                data_module.setup()
            else:
                data_module.setup_keypoints_benchmarks(stage = "test")
        else:
            data_module = DataModule(
                config_dataset, 
                dataset_class = MolfileDataset, 
                mode = "train", 
                force_precompute = precompute
            )
            data_module.setup_molfiles_benchmarks_static()

        # Read model
        model = KeypointDetector.load_from_checkpoint(
            os.path.dirname(__file__) + f"/../../../data/models/keypoint_detector/{keypoint_detector_model_name}.ckpt", 
            config_dataset = config_dataset, 
            config_training = config_training
        )

        # Set up trainer
        if force_cpu:
            trainer = pl.Trainer(
                accelerator = "cpu",
                precision = config_training["precision"],
                dlogger = False, 
            )
        else:
            trainer = pl.Trainer(
                accelerator = config_training["accelerator"],
                devices = config_training["devices"],
                precision = config_training["precision"],
                logger = False, 
            )
        
        if benchmark == "molgrapher-synthetic":
            loader = data_module.val_dataloader()
        else:
            loader = data_module.predict_dataloader()[0]
            
        predictions_out = trainer.predict(model, dataloaders=loader)
        predictions = [predictions_batch["predictions_batch"] for predictions_batch in predictions_out if predictions_batch != None]
        batches = [predictions_batch["batch"] for predictions_batch in predictions_out if predictions_batch != None]

        images = [image for batch in batches for image in batch["images"]]
        images_filenames = [image_filename for batch in batches for image_filename in batch["images_filenames"]]
        predicted_keypoints_list = [keypoints for prediction in predictions for keypoints in prediction["keypoints_batch"]]
        predicted_heatmaps = [heatmap for prediction in predictions for heatmap in prediction["heatmaps_batch"]]

        # Get bonds sizes 
        predicted_bonds_sizes = [get_bond_size(keypoints) for keypoints in predicted_keypoints_list]
        images_filenames_bonds_sizes = {k.split("/")[-1]: round(v, 2) for k, v in zip(images_filenames, predicted_bonds_sizes)}
        print("Bond sizes: ", images_filenames_bonds_sizes)
        with open(f"{config_dataset['benchmarks'][0]}_bonds_sizes.json", "w") as outfile:
            json.dump(images_filenames_bonds_sizes, outfile)

        if visualize or evaluate:
            # Rescale
            gt_keypoints_list = [gt_keypoints for batch in batches for gt_keypoints in batch["keypoints_batch"]]
           
            scaling_factor = config_dataset["image_size"][1]//config_dataset["mask_size"][1]
            gt_keypoints_list = [
                [[gt_keypoint[0]*scaling_factor + scaling_factor//2, gt_keypoint[1]*scaling_factor + scaling_factor//2
                    ] for gt_keypoint in gt_keypoints
                ] for gt_keypoints in gt_keypoints_list 
            ]
            predicted_keypoints_list = [
                [[predicted_keypoint[0]*scaling_factor + scaling_factor//2, predicted_keypoint[1]*scaling_factor + scaling_factor//2
                    ] for predicted_keypoint in predicted_keypoints
                ] for predicted_keypoints in predicted_keypoints_list 
            ]
            t = transforms.Compose([transforms.Resize([1024, 1024])])
            predicted_heatmaps = [t(heatmap) for heatmap in predicted_heatmaps]
            
        if visualize:
            # Save predictions
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            for predicted_keypoints, gt_keypoints, image, image_filename, predicted_heatmap in tqdm(zip(predicted_keypoints_list, gt_keypoints_list, images, images_filenames, predicted_heatmaps), total=len(images)):
                image = image.permute(1, 2, 0).numpy()[:, :, 0]
                #heatmap = torch.nn.Threshold(0.99999, 255)(heatmap)
                figure, axis = plt.subplots(1, 2, figsize=(14, 6))
                axis[0].imshow(image, cmap='gray') 
                axis[0].scatter(
                    *zip(*gt_keypoints),
                    color="red",
                    alpha=0.5
                )
                axis[1].imshow(predicted_heatmap.permute(1, 2, 0))
                
                axis[1].imshow(1 - image, alpha=0.25*(image < 1), cmap='gray')
                axis[1].scatter(
                    *zip(*predicted_keypoints),
                    color="white",
                    s=2,
                    alpha=0.5
                )
                plt.savefig(f"{output_folder_path}/{image_filename.split('/')[-1]}")
                plt.close()
        
        if evaluate:
            scores = defaultdict(dict)
            for gt_keypoints, predicted_keypoints, image_filename in zip(gt_keypoints_list, predicted_keypoints_list, images_filenames):
                precision, recall = get_metrics_keypoint_detector(gt_keypoints, predicted_keypoints)
                if precision == None:
                    continue
                image_name = image_filename.split('/')[-1][:-4]
                scores[image_name]["precision"] = precision
                scores[image_name]["recall"] = recall
            print(f"Scores: {scores}")
            
            nb_correct_predictions = 0
            for image_name in scores.keys():
                if scores[image_name]["precision"] == 1 and scores[image_name]["recall"] == 1:
                    nb_correct_predictions += 1
            print(f"Number of correct predictions: {nb_correct_predictions}")
            print(f"Number of test samples: {len(scores)}")
            print(f"Keypoints detection accuracy: {nb_correct_predictions/len(scores)}")

            precisions = [scores[image_name]["precision"] for image_name in scores]
            recalls = [scores[image_name]["recall"] for image_name in scores]
            print("Average precision: ", np.mean(precisions))
            print("Average recall: ", np.mean(recalls))

            with open(os.path.dirname(__file__) + f"/../../../data/scores/keypoint_detector/{keypoint_detector_model_name}_{benchmark}.json", "w") as file:
                json.dump(scores, file)

        if visualize_outliers:
            # Save predictions
            if not os.path.exists(output_folder_path + "/outliers/"):
                os.makedirs(output_folder_path + "/outliers/")

            for predicted_keypoints, gt_keypoints, image, image_filename, predicted_heatmap in tqdm(zip(predicted_keypoints_list, gt_keypoints_list, images, images_filenames, predicted_heatmaps), total=len(images)):
                image_name = image_filename.split('/')[-1][:-4]
                if image_name not in scores:
                    continue
                score = scores[image_name]
                
                # Select molecules with wrong predictions
                if (score["precision"] == 1) and (score["recall"] == 1):
                    continue

                image = image.permute(1, 2, 0).numpy()[:, :, 0]
                figure, axis = plt.subplots(1, 2, figsize=(14, 6))
                axis[0].imshow(image, cmap='gray') 
                axis[0].scatter(
                    *zip(*gt_keypoints),
                    color="red",
                    alpha=0.5
                )

                axis[1].imshow(predicted_heatmap.permute(1, 2, 0))
                axis[1].imshow(1 - image, alpha=0.25*(image < 1), cmap='gray')
                axis[1].scatter(
                    *zip(*predicted_keypoints),
                    color="white",
                    s=2,
                    alpha=0.5
                )
                plt.title(f"{score}")
                plt.savefig(f"{output_folder_path}/outliers/{image_filename.split('/')[-1]}")
                plt.close()
                    
if __name__ == "__main__":
    main()
    