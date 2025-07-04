#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import pickle
import shutil
from collections import Counter
from math import *
from pprint import pprint
from time import time

import cv2
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from mol_depict.molfile_parser.label_molfile import LabelMolFile
from mol_depict.utils.utils_drawing import draw_molecule_rdkit
from mol_depict.utils.utils_generation import get_abbreviations_smiles_mapping
from rdkit import Chem
from torchvision import transforms
from tqdm import tqdm

from molgrapher.data_modules.data_module import DataModule
from molgrapher.datasets.dataset_graph import GraphDataset
from molgrapher.datasets.dataset_image import ImageDataset
from molgrapher.datasets.dataset_molfile import MolfileDataset
from molgrapher.models.abbreviation_detector import (AbbreviationDetectorCPU,
                                                     AbbreviationDetectorGPU,
                                                     SpellingCorrector)
from molgrapher.models.graph_recognizer import (GraphRecognizer,
                                                StereochemistryRecognizer)
from molgrapher.utils.utils_dataset import get_bonds_sizes
from molgrapher.utils.utils_evaluation import (
    compute_molecule_prediction_quality, get_molecule_information)

os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-classifier-model-name", type=str, default="")
    parser.add_argument("--keypoint-detector-model-name", type=str, default="")
    parser.add_argument("--evaluation-save-path", type=str, default="")
    parser.add_argument(
        "--benchmarks", nargs="+", help="<Required> Set flag", default=[]
    )
    parser.add_argument(
        "--nb-sample-benchmarks", nargs="+", help="<Required> Set flag", default=[]
    )
    args = parser.parse_args()
    args.nb_sample_benchmarks = [int(n) for n in args.nb_sample_benchmarks]

    # Configuration
    model_name = "default"
    # Hardware
    mode = "null"  # "ddp"
    nb_ddp_processes = 1
    force_cpu = False
    # Action
    precompute = False
    evaluate = True
    visualize_outliers = False
    visualize = False
    visualize_all_steps = False
    rdkit_output = False
    evaluate_graph_classifier = False
    evaluate_superatom_ocr = False
    test_time_augmentation = False
    # DataLoader
    static_validation = True
    clean_only = False
    filtered_evaluation = False
    filtered_no_charges_evaluation = False
    preprocess = True
    # Save
    saved_predictions = False
    saved_abbreviations = False
    save_predictions = False  # Set to False for large datasets if using limited RAM
    save_abbreviations = False
    clean = True
    images_only = False
    images_folder_path = os.path.dirname(__file__) + "/../../../data/benchmarks/lum_1/"

    # Read config file
    with open(
        os.path.dirname(__file__) + "/../../../data/config_dataset_graph.json"
    ) as file:
        config_dataset_graph = json.load(file)
    with open(
        os.path.dirname(__file__) + "/../../../data/config_training_graph.json"
    ) as file:
        config_training_graph = json.load(file)
    with open(
        os.path.dirname(__file__) + "/../../../data/config_dataset_keypoint.json"
    ) as file:
        config_dataset_keypoint = json.load(file)
    with open(
        os.path.dirname(__file__) + "/../../../data/config_training_keypoint.json"
    ) as file:
        config_training_keypoint = json.load(file)
    if args.benchmarks != []:
        config_dataset_graph["benchmarks"] = args.benchmarks
    if args.nb_sample_benchmarks != []:
        config_dataset_graph["nb_sample_benchmarks"] = args.nb_sample_benchmarks

    # Sanity check
    if len(config_training_graph["devices"]) != 1:
        print("Warning the evaluation can only run on single GPU")
        exit(0)
    if len(config_dataset_graph["benchmarks"]) != len(
        config_dataset_graph["nb_sample_benchmarks"]
    ):
        print("Warning number of samples for each benchmark missing.")
        exit(0)

    starting_time = time()
    scores_logging = {}
    for benchmark, nb_sample_benchmark in zip(
        config_dataset_graph["benchmarks"], config_dataset_graph["nb_sample_benchmarks"]
    ):
        output_folder_path = (
            os.path.dirname(__file__)
            + f"/../../../data/visualization/predictions/molecular_recognition/{benchmark}_{config_dataset_graph['experiment_name']}_{model_name}"
        )
        print("Output folder path: ", output_folder_path)
        if clean and os.path.exists(output_folder_path):
            shutil.rmtree(output_folder_path)
        if not os.path.exists(output_folder_path):
            print("Output folder created")
            os.makedirs(output_folder_path)

        if test_time_augmentation:
            predictions_taa = []
            number_taa_steps = 2
        else:
            number_taa_steps = 1
        for taa_step in range(number_taa_steps):
            # Read  dataset graphs
            if evaluate_graph_classifier:
                data_module = DataModule(
                    config_dataset_graph,
                    dataset_class=GraphDataset,
                    dataset_evaluate=True,
                    clean_only=clean_only,
                )
            # Read  dataset molfiles
            elif not images_only:
                data_module = DataModule(
                    config_dataset_graph,
                    dataset_class=MolfileDataset,
                    clean_only=clean_only,
                    taa_step=taa_step,
                )
                if precompute:
                    data_module.precompute_keypoints_benchmarks(nb_sample_benchmark)

                if static_validation:
                    data_module.setup_molfiles_benchmarks_static(
                        benchmark, nb_sample_benchmark
                    )
                else:
                    data_module.setup_keypoints_benchmarks()

            # Read dataset images
            elif images_only:
                data_module = DataModule(
                    config_dataset_graph,
                    dataset_class=ImageDataset,
                    images_or_paths=images_folder_path,
                )
                data_module.setup_images_benchmarks()

            if preprocess and (benchmark != "molgrapher-synthetic"):
                print(f"Starting Caption Removal Preprocessing")
                ref_t = time()
                data_module.preprocess()
                print(
                    f"Caption Removal Preprocessing completed in {round(time() - ref_t, 2)}"
                )

            # Read model
            model = GraphRecognizer(
                config_dataset_keypoint,
                config_training_keypoint,
                config_dataset_graph,
                config_training_graph,
                keypoint_detector_model_name=args.keypoint_detector_model_name,
                graph_classifier_model_name=args.graph_classifier_model_name,
            )

            # Set up trainer
            if force_cpu:
                trainer = pl.Trainer(
                    accelerator="cpu",
                    precision=config_training_graph["precision"],
                    logger=False,
                )
            else:
                trainer = pl.Trainer(
                    accelerator=config_training_graph["accelerator"],
                    devices=config_training_graph["devices"],
                    precision=config_training_graph["precision"],
                    logger=False,
                )

            # Read loader
            if benchmark == "molgrapher-synthetic":
                data_module.setup()
                loader = data_module.val_dataloader()
            else:
                loader = data_module.predict_dataloader()

            # Get predictions
            torch.set_num_threads(config_dataset_graph["num_threads_pytorch"])
            print(f"Starting Keypoint Detection + Node Classification")
            ref_t = time()
            if not saved_predictions:
                predictions_out = trainer.predict(model, dataloaders=loader)
                if save_predictions:
                    with open(
                        os.path.dirname(__file__)
                        + f"/../../../data/predictions/predictions_{config_dataset_graph['experiment_name']}_{benchmark}_{nb_sample_benchmark}.pickle",
                        "wb",
                    ) as f:
                        pickle.dump(
                            predictions_out, f, protocol=pickle.HIGHEST_PROTOCOL
                        )
            else:
                with open(
                    os.path.dirname(__file__)
                    + f"/../../../data/predictions/predictions_{config_dataset_graph['experiment_name']}_{benchmark}_{nb_sample_benchmark}.pickle",
                    "rb",
                ) as f:
                    predictions_out = pickle.load(f)
            print(
                f"Keypoint Detection + Node Classification completed in {round(time() - ref_t, 2)}"
            )

            images_filenames = []
            images = []
            gt_graphs = []
            molfiles_filenames = []
            predictions = {"graphs": [], "keypoints": [], "confidences": []}
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
                if evaluate_graph_classifier:
                    for _elem in _prediction["batch"]["graph"]:
                        gt_graphs.append(_elem)
                if evaluate or evaluate_superatom_ocr:
                    for _elem in _prediction["batch"]["molfiles_filenames"]:
                        molfiles_filenames.append(_elem)

            if test_time_augmentation:
                predictions_taa.append(predictions)

        if test_time_augmentation:
            # Reduce predictions
            predictions_reduced = {"graphs": [], "keypoints": [], "confidences": []}
            for i in range(len(images_filenames)):
                confidences_predictions_i = [
                    predictions_taa[taa_step]["confidences"][i]
                    for taa_step in range(number_taa_steps)
                ]
                best_candidate_index = confidences_predictions_i.index(
                    max(confidences_predictions_i)
                )
                predictions_reduced["graphs"].append(
                    predictions_taa[best_candidate_index]["graphs"][i]
                )
                predictions_reduced["keypoints"].append(
                    predictions_taa[best_candidate_index]["keypoints"][i]
                )
                predictions_reduced["confidences"].append(
                    predictions_taa[best_candidate_index]["confidences"][i]
                )
            predictions = predictions_reduced

        scaling_factor = (
            config_dataset_keypoint["image_size"][1]
            / config_dataset_keypoint["mask_size"][1]
        )

        if evaluate_graph_classifier:
            correct_node_prediction = 0
            total_node_prediction = 0
            for predicted_graph, gt_graph in zip(predictions["graphs"], gt_graphs):
                predicted_keypoints = []
                predicted_classes = []
                gt_keypoints = []
                gt_classes = []
                for predicted_atom, gt_atom in zip(
                    predicted_graph.atoms, gt_graph.atoms
                ):
                    predicted_keypoints.append(predicted_atom["position"])
                    predicted_classes.append(predicted_atom["class"])
                    gt_keypoints.append(gt_atom["position"])
                    gt_classes.append(gt_atom["class"])
                if Counter(gt_classes) == Counter(predicted_classes):
                    correct_node_prediction += 1
                total_node_prediction += 1
            print(
                f"Node Classifier Molecule Precision: {round(correct_node_prediction/total_node_prediction, 2)}"
            )

        if (
            visualize
            or evaluate
            or (visualize_all_steps and rdkit_output)
            or evaluate_superatom_ocr
        ):
            # Compute bond size
            bonds_sizes = get_bonds_sizes(predictions["keypoints"], scaling_factor)

            # Recognize abbreviations
            print(f"Starting Abbreviation Recognition")
            ref_t = time()
            abbreviation_detector = AbbreviationDetectorGPU(config_dataset_graph)
            abbreviations_list = abbreviation_detector.mp_run(
                images_filenames, predictions["graphs"], bonds_sizes
            )
            print(f"Abbreviation Recognition completed in {round(time() - ref_t, 2)}")

            # Recognize stereochemistry
            print(f"Starting Stereochemistry Recognition")
            ref_t = time()
            stereochemistry_recognizer = StereochemistryRecognizer(config_dataset_graph)
            predictions["graphs"] = stereochemistry_recognizer(
                images, predictions["graphs"], bonds_sizes
            )
            print(
                f"Stereochemistry Recognition completed in {round(time() - ref_t, 2)}"
            )

            # Create RDKit graph
            print("Starting Graph creation")
            ref_t = time()
            with open(
                os.path.dirname(__file__)
                + "/../../../data/ocr_mapping/ocr_atoms_classes_mapping.json"
            ) as file:
                ocr_atoms_classes_mapping = json.load(file)
            abbreviations_smiles_mapping = get_abbreviations_smiles_mapping(
                benchmark_dataset=benchmark, filtered_evaluation=filtered_evaluation
            )
            predicted_molecules = []
            for abbreviations, graph in zip(abbreviations_list, predictions["graphs"]):
                predicted_molecule = graph.to_rdkit(
                    abbreviations,
                    abbreviations_smiles_mapping,
                    ocr_atoms_classes_mapping,
                    SpellingCorrector(abbreviations_smiles_mapping),
                    assign_stereo=False,
                    align_rdkit_output=False,
                )
                predicted_molecules.append(predicted_molecule)
            print(f"Graph creation completed in {round(time() - ref_t, 2)}")
            predictions["molecules"] = predicted_molecules

        if visualize_all_steps:
            t = transforms.Compose([transforms.Resize([1024, 1024])])
            predictions["heatmaps"] = [
                heatmaps
                for prediction in predictions_list
                for heatmaps in prediction["heatmaps_batch"]
            ]
            predictions["heatmaps"] = [
                t(heatmap) for heatmap in predictions["heatmaps"]
            ]

            if rdkit_output:
                for (
                    image_filename,
                    image,
                    graph,
                    keypoints,
                    predicted_heatmap,
                    molecule,
                ) in tqdm(
                    zip(
                        images_filenames,
                        images,
                        predictions["graphs"],
                        predictions["keypoints"],
                        predictions["heatmaps"],
                        predictions["molecules"],
                    ),
                    total=len(images_filenames),
                ):

                    figure, axis = plt.subplots(1, 1, figsize=(10, 10))
                    plt.imshow(predicted_heatmap.permute(1, 2, 0))
                    image = image.permute(1, 2, 0).numpy()[:, :, 0]
                    plt.imshow(1 - image, alpha=0.25 * (image < 1), cmap="gray")
                    plt.savefig(
                        f"{output_folder_path}/{image_filename.split('/')[-1][:-4]}-1.png",
                        dpi=500,
                    )
                    plt.close()

                    figure, axis = plt.subplots(1, 1, figsize=(7, 7))
                    graph.display_data_nodes_only(
                        axis=axis, simple_display=True, supergraph=False
                    )
                    plt.savefig(
                        f"{output_folder_path}/{image_filename.split('/')[-1][:-4]}-2.png",
                        dpi=500,
                    )
                    plt.close()

                    figure, axis = plt.subplots(1, 1, figsize=(10, 10))
                    image = draw_molecule_rdkit(
                        smiles="", molecule=molecule, augmentations=False
                    )
                    if image is not None:
                        axis.imshow(image.permute(1, 2, 0))
                    plt.savefig(
                        f"{output_folder_path}/{image_filename.split('/')[-1][:-4]}-4.png",
                        dpi=500,
                    )
                    plt.close()
            else:
                for image_filename, image, graph, keypoints, predicted_heatmap in tqdm(
                    zip(
                        images_filenames,
                        images,
                        predictions["graphs"],
                        predictions["keypoints"],
                        predictions["heatmaps"],
                    ),
                    total=len(images_filenames),
                ):

                    figure, axis = plt.subplots(1, 1, figsize=(8, 8))
                    plt.imshow(predicted_heatmap.permute(1, 2, 0))
                    image = image.permute(1, 2, 0).numpy()[:, :, 0]
                    plt.imshow(1 - image, alpha=0.25 * (image < 1), cmap="gray")
                    plt.savefig(
                        f"{output_folder_path}/{image_filename.split('/')[-1][:-4]}-1.png",
                        dpi=500,
                    )
                    plt.close()

                    figure, axis = plt.subplots(1, 1, figsize=(11, 10))
                    graph.display_data_nodes_only(
                        axis=axis, simple_display=True, supergraph=False
                    )
                    plt.savefig(
                        f"{output_folder_path}/{image_filename.split('/')[-1][:-4]}-2.png",
                        dpi=500,
                    )
                    plt.close()

                    figure, axis = plt.subplots(1, 1, figsize=(8, 8))
                    graph.display_data_nodes_only(
                        axis=axis, simple_display=False, supergraph=True
                    )
                    plt.savefig(
                        f"{output_folder_path}/{image_filename.split('/')[-1][:-4]}-3.png",
                        dpi=500,
                    )
                    plt.close()

        # Visualize predictions
        if visualize:
            for image_filename, image, graph, keypoints, molecule in tqdm(
                zip(
                    images_filenames,
                    images,
                    predictions["graphs"],
                    predictions["keypoints"],
                    predictions["molecules"],
                ),
                total=len(images_filenames),
            ):

                figure, axis = plt.subplots(1, 3, figsize=(20, 10))
                axis[0].imshow(image.permute(1, 2, 0))

                axis[0].scatter(
                    [
                        (keypoint[0] * scaling_factor + scaling_factor // 2)
                        for keypoint in keypoints
                    ],
                    [
                        (keypoint[1] * scaling_factor + scaling_factor // 2)
                        for keypoint in keypoints
                    ],
                    color="red",
                    alpha=0.5,
                )
                axis[0].title.set_text("Input Image")

                graph.display_data_nodes_only(axis=axis[1])
                axis[1].title.set_text("Predicted Graph")

                image = draw_molecule_rdkit(
                    smiles=Chem.MolToSmiles(molecule),
                    molecule=molecule,
                    augmentations=False,
                )
                if image is not None:
                    axis[2].imshow(image.permute(1, 2, 0))
                    axis[2].title.set_text("Predicted Molecule")

                plt.savefig(f"{output_folder_path}/{image_filename.split('/')[-1]}")
                plt.close()

        # Evaluate performances
        if evaluate:
            scores = {}
            molecular_information = {}
            detected_errors = 0
            for image_filename, predicted_molecule, molfile_filename in tqdm(
                zip(images_filenames, predictions["molecules"], molfiles_filenames),
                total=len(images_filenames),
            ):
                image_name = image_filename.split("/")[-1][:-4]
                gt_molecule = Chem.MolFromMolFile(molfile_filename)
                try:
                    Chem.SanitizeMol(gt_molecule)
                except:
                    print(f"Invalid MolFile: {molfile_filename}")
                    continue

                if not gt_molecule:
                    print(f"Invalid MolFile: {molfile_filename}")
                    continue
                if "PatCID-1K" in benchmark:
                    if Chem.MolToSmiles(gt_molecule, kekuleSmiles=False) == "":
                        print(f"Invalid annotation: {molfile_filename}")
                        continue

                if filtered_evaluation and ("jpo" in benchmark):
                    print("JPO evaluation can not be filtered")

                if filtered_evaluation and ("jpo" not in benchmark):
                    molfile_annotator = LabelMolFile(molfile_filename)
                    if molfile_annotator is None:
                        print(
                            f"The MolFile annotation is not supported: {molfile_filename}"
                        )
                        continue

                    are_markush, are_molecules, are_abbreviated, nb_structures = (
                        molfile_annotator.are_markush(),
                        molfile_annotator.are_molecules(),
                        molfile_annotator.are_abbreviated(),
                        molfile_annotator.how_many_structures(),
                    )

                    if (
                        not any(are_markush)
                        and not any(are_abbreviated)
                        and not any(are_molecules)
                    ):
                        print(
                            f"The MolFile annotation is not supported: {molfile_filename}"
                        )
                        continue

                    if are_markush != [False] or (nb_structures != 1):
                        print(
                            f"The MolFile contains multiple compounds or a markush structure: {molfile_filename}"
                        )
                        continue

                if filtered_no_charges_evaluation:
                    for atom in gt_molecule.GetAtoms():
                        atom.SetFormalCharge(0)

                if filtered_no_charges_evaluation and ("jpo" not in benchmark):
                    molfile_annotator = LabelMolFile(molfile_filename)
                    if molfile_annotator is None:
                        print(
                            f"The MolFile annotation is not supported: {molfile_filename}"
                        )
                        continue

                    try:
                        are_markush, are_molecules, are_abbreviated, nb_structures = (
                            molfile_annotator.are_markush(),
                            molfile_annotator.are_molecules(),
                            molfile_annotator.are_abbreviated(),
                            molfile_annotator.how_many_structures(),
                        )
                    except:
                        continue

                    if (
                        not any(are_markush)
                        and not any(are_abbreviated)
                        and not any(are_molecules)
                    ):
                        print(
                            f"The MolFile annotation is not supported: {molfile_filename}"
                        )
                        continue

                    if (
                        (are_markush != [False])
                        or (nb_structures != 1)
                        or (are_abbreviated != [False])
                        or (are_molecules != [True])
                    ):
                        print(
                            f"The MolFile contains multiple compounds or a markush structure: {molfile_filename}"
                        )
                        continue

                molecular_information[image_name] = get_molecule_information(
                    molfile_filename=molfile_filename
                )

                scores[image_name] = compute_molecule_prediction_quality(
                    Chem.MolToSmiles(predicted_molecule),
                    Chem.MolToSmiles(gt_molecule),
                    predicted_molecule=predicted_molecule,
                    gt_molecule=gt_molecule,
                    remove_stereo=True,
                    remove_double_bond_stereo=True,
                )

                if Chem.MolToSmiles(predicted_molecule) == "C":
                    detected_errors += 1

            try:
                detected_error_rate = round(
                    detected_errors
                    / sum([scores[i]["correct"] == False for i in scores.keys()]),
                    4,
                )
            except:
                detected_error_rate = 0
            score_correct = [
                scores[molecule_index]["correct"] for molecule_index in scores.keys()
            ]
            score_correct = sum(score_correct) / len(scores)

            scores_logging[benchmark] = {
                "keypoint_detector_model": model.keypoint_detector_model_name.split(
                    "/"
                )[-1],
                "graph_classifier_model": model.graph_classifier_model_name.split("/")[
                    -1
                ],
                "number_input_images": len(images_filenames),
                "number_processed_images": len(scores),
                "molecular_precision": round(score_correct, 4),
                "detected_error_rate": detected_error_rate,
            }

            with open(
                os.path.dirname(__file__)
                + "/../../../data/scores/molecular_recognition/scores_"
                + model_name
                + "_"
                + benchmark
                + ".json",
                "w",
            ) as outfile:
                json.dump(scores, outfile)

            with open(
                os.path.dirname(__file__)
                + "/../../../data/scores/molecular_recognition/information_"
                + model_name
                + "_"
                + benchmark
                + ".json",
                "w",
            ) as outfile:
                json.dump(molecular_information, outfile)

            print("Scores:")
            pprint(scores_logging)

        if evaluate_superatom_ocr:
            superatom_total = 0
            superatom_correct = 0
            gt_superatoms_list = []
            for molfile_filename in molfiles_filenames:
                gt_superatoms = []
                molecule = Chem.MolFromMolFile(molfile_filename)
                if molecule:
                    for atom in molecule.GetAtoms():
                        alias = Chem.GetAtomAlias(atom)
                        if alias != "":
                            gt_superatoms.append(alias)

                gt_superatoms_list.append(gt_superatoms)
            predicted_superatoms_list = [
                graph.superatoms for graph in predictions["graphs"]
            ]
            print(predicted_superatoms_list)
            print(gt_superatoms_list)
            for i, (predicted_superatoms, gt_superatoms) in enumerate(
                zip(predicted_superatoms_list, gt_superatoms_list)
            ):
                if gt_superatoms != []:
                    print(f"Predicted superatoms: {predicted_superatoms}")
                    print(molfiles_filenames[i])
                    print(f"Ground truth superatoms: {gt_superatoms}")
                    if set(predicted_superatoms) == set(gt_superatoms):
                        superatom_correct += 1
                    superatom_total += 1
            if superatom_total == 0:
                print("No superatoms in the test set.")
            else:
                print(
                    f"Superatom recognition precision: {superatom_correct/superatom_total}"
                )
                print(f"Superatom proportion: {superatom_total/len(images_filenames)}")

        # Visualize outliers
        if visualize_outliers:
            if not os.path.exists(output_folder_path + "/outliers/"):
                os.makedirs(output_folder_path + "/outliers/")

            for image_filename, image, graph, keypoints, molecule in tqdm(
                zip(
                    images_filenames,
                    images,
                    predictions["graphs"],
                    predictions["keypoints"],
                    predictions["molecules"],
                ),
                total=len(images_filenames),
            ):
                image_name = image_filename.split("/")[-1][:-4]
                if image_name not in scores:
                    continue
                score = scores[image_name]

                # Select molecules with wrong predictions
                if score["correct"] == True:
                    continue

                figure, axis = plt.subplots(1, 3, figsize=(30, 10))
                axis[0].imshow(image.permute(1, 2, 0))

                axis[0].scatter(
                    [
                        (keypoint[0] * scaling_factor + scaling_factor // 2)
                        for keypoint in keypoints
                    ],
                    [
                        (keypoint[1] * scaling_factor + scaling_factor // 2)
                        for keypoint in keypoints
                    ],
                    color="red",
                    alpha=0.5,
                )

                graph.display_data_nodes_only(axis=axis[1])

                image_rdkit = draw_molecule_rdkit(
                    smiles="",
                    molecule=molecule,
                    augmentations=False,
                )
                if image_rdkit is not None:
                    axis[2].imshow(image_rdkit.permute(1, 2, 0))
                plt.title(f"Tanimoto similarity: {score['tanimoto']}")
                plt.savefig(
                    f"{output_folder_path}/outliers/{image_filename.split('/')[-1]}"
                )
                plt.close()

    # Save evaluation
    if args.evaluation_save_path == "":
        args.evaluation_save_path = (
            os.path.dirname(__file__)
            + "/../../../data/scores/molecular_recognition/synthesis_"
            + model.graph_classifier_model_name.split("/")[-1]
            + "_"
            + model.keypoint_detector_model_name.split("/")[-1]
            + "_"
            + str(config_dataset_graph["benchmarks"])
            + "_"
            + str(config_dataset_graph["nb_sample_benchmarks"])
            + ".json"
        )
    with open(args.evaluation_save_path, "w") as file:
        file.write(json.dumps(scores_logging))
        print(f"Saved evaluation: {args.evaluation_save_path}")

    print(f"Evaluation/Visualization Completed in: {round(time() - starting_time, 2)}")


if __name__ == "__main__":
    main()
