#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pytorch_lightning as pl
import mahotas
import numpy as np
import glob
from timm.scheduler import StepLRScheduler

from molgrapher.utils.utils_graph_classifier import resnet18
from molgrapher.utils.utils_dataset import get_bond_size


class WAHRLoss(nn.Module):
    def __init__(self, weighted_lambda):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.weighted_lambda = weighted_lambda

    def forward(self, predicted_heatmaps, gt_heatmaps):
        weights = (gt_heatmaps**self.weighted_lambda) * torch.abs(1 - predicted_heatmaps) \
                + (1 - gt_heatmaps**self.weighted_lambda) * torch.abs(predicted_heatmaps) 
        loss = self.mse(predicted_heatmaps, gt_heatmaps) * weights
        return torch.mean(loss)

class KeypointDetector(pl.LightningModule):
    def __init__(self, config_dataset, config_training):
        super().__init__()
        self.save_hyperparameters(config_training)
        self.config_dataset = config_dataset
        self.config_training = config_training
        self.scheduler_step = 0
        self.e2e_validation_step = 0
    
        self.feature_extractor = resnet18(
            pretrained = False, 
            output_layers = ['layer4'], 
            dilation_factor = 8,
            conv1_stride = 2
        )
        self.nb_filters = list(self.feature_extractor.children())[-3][-1].conv1.out_channels

        self.conv1 = nn.Conv2d(in_channels=self.nb_filters, out_channels=self.nb_filters//2, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(self.nb_filters//2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=self.nb_filters//2, out_channels=1, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.criterion = nn.MSELoss() # WAHRLoss()

    def forward(self, x):
        x = self.feature_extractor(x)["layer4"]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x
    
    def training_step(self, batch):
        predicted_heatmaps = self.forward(batch["images"])
        loss = self.criterion(predicted_heatmaps, batch["heatmaps_batch"])
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=self.config_dataset["batch_size"])
        return loss
    
    def validation_step(self, batch, batch_idx):
        predicted_heatmaps = self.forward(batch["images"])
        loss = self.criterion(predicted_heatmaps, batch["heatmaps_batch"])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=self.config_dataset["batch_size"])
        return loss
  
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config_training["lr"]) #AdamW
        scheduler = StepLRScheduler(
            optimizer, 
            decay_t=self.config_training["decay_t"],
            decay_rate=self.config_training["decay_rate"]
        )

        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step",
                "frequency": self.config_training["decay_step_frequency"]
            }
        }

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        if scheduler.optimizer.param_groups[0]["lr"] > 1e-6:
            # The custom LR Scheduler requires to overwrite lr_scheduler_step.
            self.scheduler_step += 1
        scheduler.step(epoch = self.scheduler_step)
    
    def predict(self, images):
        #self.threshold = 1#55
        #ref_t = time()

        predicted_heatmaps = self.forward(images)
        #print(f"Keypoints detection (heatmap) completed in {round(time() - ref_t, 3)}")
        #ref_t = time()

        predicted_heatmaps = predicted_heatmaps.cpu().detach()
        #print(f"Keypoints detection (conversion) completed in {round(time() - ref_t, 3)}") # 50% of time
        #ref_t = time()

        keypoints_batch = []
        for heatmap in predicted_heatmaps:
            #counts, bins = np.histogram([item for sublist in heatmap.tolist() for item in sublist], bins = 100)
            #heatmap = nn.Threshold(list(bins)[-self.threshold], 0)(heatmap)
            heatmap_thresholded = nn.Threshold(0.1, 0)(heatmap)
            Bc = np.ones((5, 5))
            heatmap_max = mahotas.regmax(heatmap_thresholded.numpy()[0, :, :], Bc=Bc).astype(int)
            
            keypoints_y, keypoints_x = np.where(heatmap_max == 1)
            keypoints = [[int(keypoint_x), int(keypoint_y)] for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y)]

            max_number_keypoints = 300
            if len(keypoints) > max_number_keypoints:
                print("Keypoints detection error: max number of keypoints exceeded")
                keypoints_batch.append(keypoints[:max_number_keypoints])
                continue

            if len(keypoints) == 0:
                print("Keypoints detection error: no keypoints detected")
                keypoints_batch.append(keypoints)
                continue

            if len(keypoints) == 1:
                print("Keypoints detection error: only one keypoint detected")
                keypoints_batch.append(keypoints)
                continue
            
            bond_size = get_bond_size(keypoints)
            # For small molecules, change post-processing parameters
            if bond_size > 35:
                Bc = np.ones((15, 15))
                heatmap_max = mahotas.regmax(heatmap_thresholded.cpu().detach().numpy()[0, :, :], Bc=Bc).astype(int)
                keypoints_y, keypoints_x = np.where(heatmap_max == 1)
                keypoints = [[int(keypoint_x), int(keypoint_y)] for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y)]
                
            keypoints_batch.append(keypoints)
        
        #print(f"Keypoints detection (regmax) completed in {round(time() - ref_t, 3)}") # 50% of time

        return {
            "keypoints_batch": keypoints_batch, 
            "heatmaps_batch": [predicted_heatmap for predicted_heatmap in predicted_heatmaps]
        } 
    
    def predict_step(self, batch, batch_idx, drop_last = False):
        torch.set_num_threads(self.config_dataset["num_threads_pytorch"])
        return {
            
            "predictions_batch": self.predict(batch["images"]),
            "batch": batch
        }
