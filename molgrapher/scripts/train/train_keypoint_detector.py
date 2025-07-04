#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

import cv2
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from molgrapher.data_modules.data_module import DataModule
from molgrapher.datasets.dataset_keypoint import KeypointDataset
from molgrapher.datasets.dataset_keypoint_cede import KeypointCedeDataset
from molgrapher.models.keypoint_detector import KeypointDetector

os.environ["OMP_NUM_THREADS"] = "4"  # Optimal value: nb_cpu_threads / nproc_per_node
# os.environ["NCCL_SOCKET_NTHREADS"] = "2"
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
# os.environ["MKL_NUM_THREADS"] = "2"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "4"
# os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
cv2.setNumThreads(0)
torch.set_float32_matmul_precision("medium")


def main():
    # logger = set_logging("ocsr_logger", f'{os.path.join(os.path.dirname(__file__))}/log.txt')
    mode = "train"  # "fine-tuning"
    force_precompute = False
    clean_only = False

    # Read config file
    with open(
        os.path.dirname(__file__) + "/../../../data/config_dataset_keypoint.json"
    ) as file:
        config_dataset = json.load(file)

    # Read config file
    with open(
        os.path.dirname(__file__) + "/../../../data/config_training_keypoint.json"
    ) as file:
        config_training = json.load(file)

    # Read dataset (Wrapped to a DistributedDataModule)
    if config_dataset["training_dataset"] == "molgrapher-synthetic":
        data_module = DataModule(
            config=config_dataset,
            dataset_class=KeypointDataset,
            mode=mode,
            force_precompute=force_precompute,
            clean_only=clean_only,
        )
    # data_module.setup(stage = "train")
    elif config_dataset["training_dataset"] == "cede-synthetic":
        data_module = DataModule(
            config_dataset,
            dataset_class=KeypointCedeDataset,
            mode=mode,
            force_precompute=force_precompute,
            clean_only=clean_only,
        )
    else:
        print("No Training Dataset Given")

    # Instantiate model
    if mode == "fine-tuning":
        model = KeypointDetector.load_from_checkpoint(
            os.path.dirname(__file__)
            + f"/../../../data/models/keypoint_detector/{config_training['checkpoint']}.ckpt",
            config_dataset=config_dataset,
            config_training=config_training,
        )

    else:
        model = KeypointDetector(config_dataset, config_training)

    # Set up checkpoint
    checkpoint = ModelCheckpoint(
        filename=os.path.dirname(__file__)
        + f"/../../../data/models/keypoint_detector/{config_dataset['experiment_name']}-{config_training['run_name']}-{{val_loss:.4f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Set learning monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    if mode == "fine-tuning":
        nb_sample = config_dataset["nb_sample_fine-tunning"]
    else:
        nb_sample = config_dataset["nb_sample"]
    if isinstance(config_training["devices"], list):
        nb_iterations = nb_sample / (
            config_dataset["batch_size"] * len(config_training["devices"])
        )
    else:
        nb_iterations = nb_sample / (config_dataset["batch_size"])

    tensorboard_logger = TensorBoardLogger(
        save_dir=os.path.dirname(__file__) + "/../../../data/logs/keypoint_detector/",
        name=config_dataset["experiment_name"],
        version=config_training["run_name"],
    )

    # Set up trainer
    trainer = pl.Trainer(
        accelerator=config_training["accelerator"],
        devices=config_training["devices"],
        precision=config_training["precision"],
        max_epochs=config_training["max_epochs"],
        callbacks=[checkpoint, lr_monitor],
        default_root_dir=os.path.dirname(__file__)
        + "/../../../data/logs/keypoint_detector/",
        val_check_interval=nb_iterations // 10,
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        logger=tensorboard_logger,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
