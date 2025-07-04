#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil

import matplotlib.pyplot as plt
from pytorch_lightning.trainer.states import TrainerFn
from tqdm import tqdm

from molgrapher.data_modules.data_module import DataModule
from molgrapher.datasets.dataset_graph import GraphDataset
from molgrapher.datasets.dataset_graph_cede import GraphCedeDataset


def main():
    output_folder = "test15"
    clean = True
    mode = "train"  # "fine-tuning"
    max_index = 100

    if clean:
        if os.path.exists(
            os.path.dirname(__file__)
            + f"/../../../data/visualization/training_images/graph_classifier/{output_folder}"
        ):
            shutil.rmtree(
                os.path.dirname(__file__)
                + f"/../../../data/visualization/training_images/graph_classifier/{output_folder}/"
            )
    if not os.path.exists(
        os.path.dirname(__file__)
        + f"/../../../data/visualization/training_images/graph_classifier/{output_folder}"
    ):
        os.makedirs(
            os.path.dirname(__file__)
            + f"/../../../data/visualization/training_images/graph_classifier/{output_folder}"
        )

    # Read config files
    with open(
        os.path.dirname(__file__) + "/../../../data/config_dataset_graph_2.json"
    ) as file:
        config_dataset = json.load(file)

    with open(
        os.path.dirname(__file__) + "/../../../data/config_training_graph_2.json"
    ) as file:
        config_training = json.load(file)

    config_dataset["nb_sample"] = min(config_dataset["nb_sample"], max_index)

    # Read  dataset
    data_module = DataModule(
        config_dataset, dataset_class=GraphDataset, mode=mode, force_precompute=False
    )
    # data_module.precompute_keypoints_synthetic()
    data_module.setup(stage=TrainerFn.FITTING)

    loader = data_module.train_dataloader()

    for index in tqdm(
        range(min(len(loader) * config_dataset["batch_size"], max_index))
    ):
        sample = loader.dataset.__getitem__(index, visualization=True)
        data, image, molecular_graph, image_filename = (
            sample["data"],
            sample["images"],
            sample["molecular_graph"],
            sample["images_filenames"],
        )
        figure, axis = plt.subplots(1, 2, figsize=(20, 10))

        axis[0].scatter(
            [x for x, y in data.nodes_positions],
            [y for x, y in data.nodes_positions],
            c="r",
            marker="x",
            alpha=1,
        )

        axis[0].imshow(image.permute(1, 2, 0))
        molecular_graph.display_data_nodes_only(axis=axis[1])
        plt.savefig(
            os.path.dirname(__file__)
            + f"/../../../data/visualization/training_images/graph_classifier/{output_folder}/{image_filename.split('/')[-1]}"
        )
        plt.close()


if __name__ == "__main__":
    main()
