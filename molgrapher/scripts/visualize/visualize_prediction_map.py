#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colors as mcolors
from torch_geometric.data import Batch
from tqdm import tqdm

from molgrapher.datasets.dataset_benchmark import BenchmarkDataset
from molgrapher.models.graph_classifier import GraphClassifier
from molgrapher.utils.utils_graph import MolecularGraph


def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers

    if not ax:
        ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def visualize_predictions_maps(
    path,
    graph,
    graph_classifier,
    labels_colors,
    node_type,
    positions,
    image,
    classes_labels,
    ax,
    limit,
):
    graph.set_nodes_edges_nodes_only()
    data = graph.to_torch_nodes_only()

    batch = {"data": Batch.from_data_list([data]), "images": image.unsqueeze(0)}
    graph_classifier.eval()
    with torch.no_grad():
        predictions = graph_classifier.forward(batch)
    predictions_confidences = [
        torch.max(prediction).item() for prediction in predictions[not node_type]
    ]
    predictions = [
        classes_labels[torch.argmax(prediction).item()]
        for prediction in predictions[not node_type]
    ]

    predicted_classes_positions = defaultdict(list)
    for index in range(len(positions)):
        predicted_classes_positions[predictions[index]].append(
            {"position": positions[index], "confidence": predictions_confidences[index]}
        )

    counts, bins = np.histogram(predictions_confidences, bins=100)

    confidence_threshold = list(bins)[-limit]  # TODO To get confidence, apply softmax
    labels = []
    for predicted_class in predicted_classes_positions.keys():
        positions = [
            predicted_classes_positions[predicted_class][i]["position"]
            for i in range(len(predicted_classes_positions[predicted_class]))
        ]
        confidences = [
            predicted_classes_positions[predicted_class][i]["confidence"]
            for i in range(len(predicted_classes_positions[predicted_class]))
        ]
        mscatter(
            x=[position[0] for position in positions],
            y=[position[1] for position in positions],
            color=labels_colors[predicted_class],
            alpha=0.3,
            m=[
                "o" if confidence > confidence_threshold else "."
                for confidence in confidences
            ],
            ax=ax,
        )
        labels.append(predicted_class)

    leg = ax.legend(
        labels=labels, scatterpoints=1, loc="center left", bbox_to_anchor=(1, 0.5)
    )
    for lh in leg.legendHandles:
        lh.set_alpha(1)
        lh.set_sizes([6.0])

    ax.imshow(image.permute(1, 2, 0))


def main():
    images_names = ["US10899758-20210126-C00179"]
    for image_name in images_names:
        atoms_predictions_map = True
        bonds_predictions_map = True
        resolution = 8

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
            os.path.dirname(__file__) + "/../../../data/config_dataset_benchmark.json"
        ) as file:
            config_dataset_benchmark = json.load(file)

        # Create dataset
        dataset = pd.DataFrame(
            {
                "image_filename": [
                    os.path.dirname(__file__)
                    + f"/../../../data/benchmarks/uspto_large/images/"
                    + image_name
                    + ".png"
                ],
                "molfile_filename": [
                    os.path.dirname(__file__)
                    + f"/../../../data/benchmarks/uspto_large/molfiles/"
                    + image_name
                    + ".MOL"
                ],
            }
        )
        benchmark_dataset = BenchmarkDataset(dataset, config_dataset_benchmark)
        image = benchmark_dataset[0]["images"]

        # Instantiate model
        # graph_classifier_model_name = "exp-ad-01-run-025-val_loss=0.2680"
        graph_classifier_model_name = "exp-ad-10-run-010-val_loss=0.1467"
        graph_classifier = GraphClassifier.load_from_checkpoint(
            os.path.dirname(__file__)
            + f"/../../../data/models/graph_classifier/{graph_classifier_model_name}.ckpt",
            config_dataset=config_dataset_graph,
            config_training=config_training_graph,
        )

        # Define color mapping
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()) * 50
        white_colors = [
            "beige",
            "bisque",
            "blanchedalmond",
            "cornsilk",
            "antiquewhite",
            "ghostwhite",
            "w",
            "whitesmoke",
            "white",
            "snow",
            "seashell",
            "mistyrose",
            "linen",
            "floralwhite",
            "ivory",
            "honeydew",
            "mintcream",
            "azure",
            "aliceblue",
            "lavenderblush",
        ]
        colors = [color for color in colors if color not in white_colors] * 50

        symbols_classes_atoms = json.load(
            open(
                os.path.dirname(__file__)
                + "/../../../data/vocabularies/vocabulary_atoms.json"
            )
        )
        types_classes_bonds = json.load(
            open(
                os.path.dirname(__file__)
                + "/../../../data/vocabularies/vocabulary_bonds.json"
            )
        )
        atoms_classes_symbols = {v: k for k, v in symbols_classes_atoms.items()}
        bonds_classes_types = {v: k for k, v in types_classes_bonds.items()}
        atoms_colors = {k: colors[v] for k, v in symbols_classes_atoms.items()}
        bonds_colors = {k: colors[v] for k, v in types_classes_bonds.items()}

        fig, ax = plt.subplots(1, 2, figsize=(18, 8))

        # Atoms predictions map
        if atoms_predictions_map:
            predicted_atoms = {}
            predicted_bonds = {}
            predicted_atoms_confidences = {}
            predicted_bonds_confidences = {}
            positions = []

            # Create graph
            graph = MolecularGraph(config_dataset_graph)
            for x in range(0, 1024, resolution):
                for y in range(0, 1024, resolution):
                    position = (x, y)
                    graph.atoms.append(
                        {"index": 0, "class": 0, "position": position, "type": 1}
                    )
                    positions.append(position)

            visualize_predictions_maps(
                path=os.path.dirname(__file__)
                + "/../../../data/visualization/predictions_maps/"
                + image_name
                + "_atoms_predictions.jpg",
                graph=graph,
                graph_classifier=graph_classifier,
                labels_colors=atoms_colors,
                node_type=1,
                positions=positions,
                image=image,
                classes_labels=atoms_classes_symbols,
                ax=ax[0],
                limit=10,
            )

        # Bonds predictions map
        if bonds_predictions_map:
            positions = []
            atoms_positions = []
            # Create graph
            graph = MolecularGraph(config_dataset_graph)
            atom_index = 0
            for x in range(0, 1024, resolution):
                for y in range(0, 1024, resolution):
                    position = (x, y)
                    graph.atoms.append(
                        {
                            "index": atom_index,
                            "class": 0,
                            "position": position,
                            "type": 1,
                        }
                    )

                    atom_index += 1
                    atoms_positions.append(position)

            for atom_index in range(len(graph.atoms) - 1):
                if (atom_index + 1) % (1024 // resolution) == 0:
                    continue
                graph.bonds.append(
                    {
                        "index": [atom_index, atom_index + 1],
                        "class": 0,
                        "position": [0, 0],
                        "type": 0,
                    }
                )
                positions.append(
                    [
                        (
                            atoms_positions[atom_index][0]
                            + atoms_positions[atom_index + 1][0]
                        )
                        // 2,
                        (
                            atoms_positions[atom_index][1]
                            + atoms_positions[atom_index + 1][1]
                        )
                        // 2,
                    ]
                )

            visualize_predictions_maps(
                path=os.path.dirname(__file__)
                + "/../../../data/visualization/predictions_maps/"
                + image_name
                + "_bonds_predictions.jpg",
                graph=graph,
                graph_classifier=graph_classifier,
                labels_colors=bonds_colors,
                node_type=0,
                positions=positions,
                image=image,
                classes_labels=bonds_classes_types,
                ax=ax[1],
                limit=50,
            )

        plt.savefig(
            os.path.dirname(__file__)
            + "/../../../data/visualization/predictions_maps/"
            + image_name
            + "_prediction_map.jpg"
        )


if __name__ == "__main__":
    main()
