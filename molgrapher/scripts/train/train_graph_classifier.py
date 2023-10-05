#!/usr/bin/env python3  
# -*- coding: utf-8 -*-

import cv2
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import os
import json
from pytorch_lightning.trainer.states import TrainerFn

from molgrapher.utils.utils_logging import set_logging
from molgrapher.datasets.dataset_graph import GraphDataset
from molgrapher.datasets.dataset_graph_cede import GraphCedeDataset
from molgrapher.models.graph_classifier import GraphClassifier
from molgrapher.data_modules.data_module import DataModule

os.environ["OMP_NUM_THREADS"] = "4" # Optimal value: nb_cpu_threads / nproc_per_node
#os.environ["NCCL_SOCKET_NTHREADS"] = "2"
#os.environ["OPENBLAS_NUM_THREADS"] = "4" 
#os.environ["MKL_NUM_THREADS"] = "2" 
#os.environ["VECLIB_MAXIMUM_THREADS"] = "4" 
#os.environ["NUMEXPR_NUM_THREADS"] = "4"
#os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
cv2.setNumThreads(0)
torch.set_float32_matmul_precision("medium") 


def main():
    """
    main() is called by lightning on each GPU process.
    """
    mode = "train"#"fine-tuning" 
    clean_only = False
    force_precompute = False
    precompute = False
    #logger = set_logging("ocsr_logger", f'{os.path.join(os.path.dirname(__file__))}/log.txt')

    # Read config file
    with open(os.path.dirname(__file__) + "/../../../data/config_dataset_graph_2.json") as file:
        config_dataset = json.load(file)

    with open(os.path.dirname(__file__) + "/../../../data/config_training_graph_2.json") as file:
        config_training = json.load(file)

    # Read dataset (Wrapped to a DistributedDataModule)
    if config_dataset["training_dataset"] == "molgrapher-synthetic":
        data_module = DataModule(
            config_dataset, 
            dataset_class = GraphDataset, 
            mode = mode, 
            force_precompute = force_precompute,
            clean_only = clean_only
        )
        if precompute:
            if len(config_training["devices"]) > 1:
                print("Warning: Precomputing keypoints requires only 1 ddp process.")
            data_module.precompute_keypoints_synthetic()
            exit(0)
    elif config_dataset["training_dataset"] == "cede-synthetic":
        data_module = DataModule(
            config_dataset, 
            dataset_class = GraphCedeDataset, 
            mode = mode, 
            force_precompute = force_precompute,
            clean_only = clean_only
        )
    else:
        print("No Training Dataset Given")

    # Instantiate model
    if mode == "fine-tuning":
        model = GraphClassifier.load_from_checkpoint(
            os.path.dirname(__file__) + f"/../../../data/models/graph_classifier/{config_training['checkpoint']}.ckpt", 
            config_dataset = config_dataset, 
            config_training = config_training
        )
    else:
        model = GraphClassifier(config_dataset, config_training)

    # Set up checkpoint-
    if mode == "fine-tuning":
        checkpoint = ModelCheckpoint(
            filename = f"{os.path.dirname(__file__)}/../../../data/models/graph_classifier/{config_dataset['experiment_name']}-{config_training['run_name']}-{{epoch}}-{{step}}-{{NA-R1:.3f}}-{{NA-S:.3f}}",
            save_top_k = 5, 
            monitor = "NA-R1", 
            mode = 'max'
        )
    else:
        checkpoint = ModelCheckpoint(
            filename = f"{os.path.dirname(__file__)}/../../../data/models/graph_classifier/{config_dataset['experiment_name']}-{config_training['run_name']}-{{val_loss-S:.3f}}-{{NA-S:.3f}}-{{MA-S:.3f}}",
            save_top_k = 3, 
            monitor = "val_loss-S", # Best model monitoring train_loss?
            mode = 'min'
        )

    # Set learning monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if mode == "fine-tuning":
        # Estimate 
        nb_sample = config_dataset["nb_sample_fine-tuning"]
    else:
        nb_sample = config_dataset["nb_sample"]
    if isinstance(config_training["devices"], list):
        nb_iterations = nb_sample/(config_dataset["batch_size"]*len(config_training["devices"]))
    else:
        nb_iterations = nb_sample/(config_dataset["batch_size"])
    print(f"Number of training iterations: {nb_iterations} (Four validations per epoch)")

    tensorboard_logger = TensorBoardLogger(
        save_dir = os.path.dirname(__file__) + "/../../../data/logs/graph_classifier/",
        name = config_dataset["experiment_name"],
        version = config_training["run_name"]
    )

    # Set up data module
    data_module.setup(stage = TrainerFn.FITTING)

    # Set up trainer for validation
    trainer_validation = pl.Trainer(
        accelerator = config_training["accelerator"],
        devices = [config_training["devices"][0]],
        precision = config_training["precision"],
        max_epochs = config_training["max_epochs"],
        callbacks = [checkpoint, lr_monitor],
        default_root_dir = os.path.dirname(__file__) + "/../../../data/logs/graph_classifier/",
        val_check_interval = nb_iterations//4,
        logger = tensorboard_logger
    )
    #trainer_validation.validate(model, data_module)

    # Set up trainer
    trainer = pl.Trainer(
        accelerator = config_training["accelerator"],
        devices = config_training["devices"],
        precision = config_training["precision"],
        max_epochs = config_training["max_epochs"],
        callbacks = [checkpoint, lr_monitor],
        default_root_dir = os.path.dirname(__file__) + "/../../../data/logs/graph_classifier/",
        val_check_interval = nb_iterations//4,
        strategy = DDPStrategy(find_unused_parameters=False, static_graph=True),
        logger = tensorboard_logger
    )
    # Train
    torch.set_num_threads(config_dataset["num_threads_pytorch"])
    trainer.fit(model, data_module)
   

if __name__ == "__main__":
    main()