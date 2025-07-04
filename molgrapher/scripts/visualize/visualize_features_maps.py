#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage as ndimage
import torch
from tqdm import tqdm

from molgrapher.datasets.dataset_benchmark import BenchmarkDataset
from molgrapher.models.graph_classifier import GraphClassifier


def main():
    image_name = "US20210002252A1-20210107-C00193"

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
    images = image.unsqueeze(0)

    # Instantiate model
    graph_classifier_model_name = "exp-ad-03-run-003-val_loss=0.1246"
    graph_classifier = GraphClassifier.load_from_checkpoint(
        os.path.dirname(__file__)
        + f"/../../../data/models/graph_classifier/{graph_classifier_model_name}.ckpt",
        config_dataset=config_dataset_graph,
        config_training=config_training_graph,
    )
    graph_classifier.eval()
    with torch.no_grad():
        features_maps = graph_classifier.backbone(images)["layer4"][0]

    image_size = [122, 122]
    features_maps[:, image_size[0] :, :] = 0
    features_maps[:, :, image_size[1] :] = 0
    feature_map = torch.sum(features_maps, 0)

    extent = (0, 1024, 0, 1024)
    plt.figure(figsize=(12, 12))
    image = images[0].permute(1, 2, 0).numpy()[:, :, 0]
    plt.imshow(
        feature_map.detach(), alpha=1, cmap="jet", interpolation="None", extent=extent
    )
    plt.imshow(image, alpha=0.99 * (image < 1), cmap="gray", extent=extent)
    plt.savefig(
        os.path.dirname(__file__)
        + "/../../../data/visualization/features_maps/"
        + image_name
        + "_sum.jpg"
    )

    nb_filters = [5, 5]
    figure_size = [40, 40]
    nb_plots = 80
    plot_index = 0
    feature_map_index = 0

    for plot_index in tqdm(range(nb_plots)):
        fig, ax = plt.subplots(nb_filters[0], nb_filters[1], figsize=figure_size)
        for i in range(nb_filters[0]):
            for j in range(nb_filters[1]):
                feature_map = features_maps[feature_map_index]
                # alphas = 0.8*(feature_map > 0) #+ 0.*(feature_map < 0)
                alphas = 1
                feature_map = ndimage.gaussian_filter(
                    feature_map, sigma=(2, 2), order=0
                )
                ax[i, j].imshow(feature_map, alpha=alphas, cmap="hot_r", extent=extent)
                ax[i, j].imshow(
                    image, alpha=0.99 * (image < 1), cmap="gray", extent=extent
                )
                ax[i, j].set_title(str(feature_map_index))
                feature_map_index += 1
        plt.savefig(
            os.path.dirname(__file__)
            + "/../../../data/visualization/features_maps/"
            + image_name
            + "_"
            + str(plot_index)
            + ".jpg"
        )


if __name__ == "__main__":
    main()
