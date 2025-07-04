#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl

# Modules
from molgrapher.data_modules.data_module import DataModule
from molgrapher.datasets.dataset_graph import GraphDataset
from molgrapher.models.graph_classifier import GraphClassifier


def main():
    model_name = "exp-ad-01-run-014-val_loss=0.0337"
    precompute = False
    output_folder_path = (
        os.path.dirname(__file__)
        + f"/../../../data/visualization/predictions_uspto_large_graph_classifier_aug_14/"
    )
    input_dataset = "uspto_large"

    # Read config files
    with open(
        os.path.dirname(__file__) + "/../../../data/config_dataset_graph.json"
    ) as file:
        config_dataset = json.load(file)

    with open(
        os.path.dirname(__file__) + "/../../../data/config_training_graph.json"
    ) as file:
        config_training = json.load(file)

    # Read  dataset
    data_module = DataModule(config_dataset, dataset_class=GraphDataset)
    if precompute:
        data_module.precompute_keypoints_benchmarks()
        data_module.precompute_keypoints_synthetic()
    data_module.setup_keypoints_benchmarks()
    data_module.setup()

    # Read model
    model = GraphClassifier.load_from_checkpoint(
        os.path.dirname(__file__)
        + f"/../../../data/models/graph_classifier/{model_name}.ckpt",
        config_dataset=config_dataset,
        config_training=config_training,
        strict=False,
    )

    # Set up trainer
    trainer = pl.Trainer(
        accelerator=config_training["accelerator"],
        devices=config_training["devices"],
        precision=config_training["precision"],
        logger=False,
    )

    loader = data_module.predict_dataloader(input_dataset)
    # loader = data_module.val_dataloader()
    images = [image for batch in loader for image in batch["images"]]
    gt_graphs = []
    for batch in loader:
        for batch_index in range(config_dataset["batch_size"]):
            gt_graphs.append(batch["data"].get_example(batch_index))

    predictions = trainer.predict(model, dataloaders=loader)
    predictions = [
        predictions_batch
        for predictions_batch in predictions
        if predictions_batch != None
    ]
    predictions = [
        prediction
        for predictions_batch in predictions
        for prediction in predictions_batch
    ]

    # Save predictions
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for index, (image, graph, gt_graph) in enumerate(
        zip(images, predictions, gt_graphs)
    ):
        figure, axis = plt.subplots(1, 2, figsize=(20, 10))
        axis[0].imshow(image.permute(1, 2, 0))

        axis[0].scatter(
            [position[0] for position in gt_graph.nodes_positions],
            [position[1] for position in gt_graph.nodes_positions],
            color="red",
            alpha=0.5,
        )

        graph.display_data_nodes_only(axis=axis[1])
        plt.savefig(f"{output_folder_path}/{index}.png")
        plt.close()


if __name__ == "__main__":
    main()
