import json
import os
import glob
import subprocess
from collections import defaultdict
import numpy as np
import torch
from pprint import pprint
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from timm.scheduler import StepLRScheduler
from torch_geometric.nn import GCNConv

from molgrapher.utils.utils_graph_classifier import get_point_of_trisection_coordinates, get_positional_encoding_2d, resnet50
from molgrapher.utils.utils_graph import MolecularGraph


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
      
        self.backbone = resnet50(
            pretrained = False, 
            output_layers = ['layer4'], 
            dilation_factor = 2,
            conv1_stride = 2
        )
        self.nb_filters = 2048
        self.features_maps_spatial_size = [64, 64]

        
    def forward(self, x):
        # (N, 3, 1024, 1024)
        #print(self.backbone[-1][-1].conv1.weight[0]) # Debug
        return self.backbone(x)

class MessagePassing(pl.LightningModule):
    def __init__(self, node_embedding_dimension):
        super().__init__()
        self.node_embedding_dimension = node_embedding_dimension
        self.max_nb_neighbors = 100
        self.mlp = nn.Sequential(
            nn.Linear(self.max_nb_neighbors, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        global_consistency = torch.ones(data.num_nodes, device=self.device)

        # Get atoms neighboring bonds indices (Works for bipartite graph)
        neighboring_bonds_indices = defaultdict(list)
        for index_b, index_e in data.edge_index.t():
            if data.nodes_types[index_b.item()].item() == 1: 
                if index_e.item() not in neighboring_bonds_indices[index_b.item()]:
                    neighboring_bonds_indices[index_b.item()].append(index_e.item())
            if data.nodes_types[index_e.item()].item() == 1: 
                if index_b.item() not in neighboring_bonds_indices[index_e.item()]:
                    neighboring_bonds_indices[index_e.item()].append(index_b.item())

        for node_index, (central_node_embedding, node_type) in enumerate(zip(data.x, data.nodes_types)):
            if node_type == 1:
                neighbors_nodes_embeddings = data.x[neighboring_bonds_indices[node_index]]

                # Zero-padding
                if neighbors_nodes_embeddings.size(0) < self.max_nb_neighbors:
                    neighbors_nodes_embeddings = torch.cat((
                        neighbors_nodes_embeddings, 
                        torch.zeros(self.max_nb_neighbors - neighbors_nodes_embeddings.size(0), self.node_embedding_dimension, device=self.device)
                    ))
                neighbors_nodes_embeddings = neighbors_nodes_embeddings.permute(1, 0)
                constraint_embedding = self.mlp(neighbors_nodes_embeddings)[:, 0]
                global_consistency[node_index] = torch.dot(constraint_embedding, central_node_embedding)/(torch.norm(constraint_embedding)*torch.norm(central_node_embedding))
                
        return global_consistency

class GNN(nn.Module):
    def __init__(self, config_dataset, node_embedding_dimension, device):
        super().__init__()
        self.config_dataset = config_dataset
        self.node_embedding_dimension = node_embedding_dimension
        self.nb_filters_out = node_embedding_dimension

        self.gcn_on = False

        if self.gcn_on:
            # graph_classifier_model_name = "exp-ad-11-run-mp-4-64-02-val_loss=0.0086"
            self.nb_filters_out = node_embedding_dimension//8
            self.conv1 = GCNConv(self.node_embedding_dimension, self.node_embedding_dimension//4)
            self.conv2 = GCNConv(self.node_embedding_dimension//4, self.node_embedding_dimension//8)
            self.conv3 = GCNConv(self.node_embedding_dimension//8, self.node_embedding_dimension//8)
            self.conv4 = GCNConv(self.node_embedding_dimension//8, self.nb_filters_out)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p = 0.2)
        
        # Classifier
        self.mlp_atoms = nn.Sequential(
            nn.Linear(self.nb_filters_out, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.15), 
            nn.Linear(256, self.config_dataset["nb_atoms_classes"])
        )

        self.mlp_bonds = nn.Sequential(
            nn.Linear(self.nb_filters_out, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.15),
            nn.Linear(256, self.config_dataset["nb_bonds_classes"])
        )
        
    
    def forward(self, data):
        if self.gcn_on:
            data.x = self.conv1(data.x, data.edge_index)
            data.x = self.relu(data.x)
            data.x = self.dropout(data.x)
            data.x = self.conv2(data.x, data.edge_index)
            data.x = self.relu(data.x)
            data.x = self.dropout(data.x)
            data.x = self.conv3(data.x, data.edge_index)
            data.x = self.relu(data.x)
            data.x = self.dropout(data.x)
            data.x = self.conv4(data.x, data.edge_index)
            data.x = self.relu(data.x)
            data.x = self.dropout(data.x)

        # data.x [nb_nodes, nb_filters]
        atoms_indices = [index for index in range(len(data.nodes_types)) if (data.nodes_types[index] == 1)]
        bonds_indices = [index for index in range(len(data.nodes_types)) if (data.nodes_types[index] == 0)]
        
        atoms_predictions = self.mlp_atoms(data.x[atoms_indices])
        bonds_predictions = self.mlp_bonds(data.x[bonds_indices])
        
        return [atoms_predictions, bonds_predictions]


class GraphClassifier(pl.LightningModule):
    def __init__(self, config_dataset, config_training):
        super().__init__()
        self.save_hyperparameters(config_training)
        self.config_dataset = config_dataset
        self.config_training = config_training
        self.scheduler_step = 0 
        self.e2e_validation_step = 0

        # Parameters
        self.backbone = Backbone()
        self.node_embedding_dimension = self.backbone.nb_filters 
        self.gnn = GNN(config_dataset, self.node_embedding_dimension, self.device)
        
        self.bond_nb_sampled_points = 3 #5
        self.bond_fusion = False

        if self.bond_fusion:
            self.bond_fusor = nn.Sequential(
                nn.Linear(self.bond_nb_sampled_points, 1024), 
                nn.ReLU(),
                nn.Dropout(p = 0.2),
                nn.Linear(1024, 1) 
            )
        
        self.positional_embedding = False
        if self.positional_embedding:
            self.register_buffer("node_positional_embedding_map", 
                get_positional_encoding_2d(
                    self.node_embedding_dimension, 
                    height = self.backbone.features_maps_spatial_size[0], 
                    width = self.backbone.features_maps_spatial_size[1]
                )
            )

        self.type_embedding = True
        if self.type_embedding:
            self.atom_type_embedding = nn.Parameter(torch.randn(self.node_embedding_dimension))
            self.bond_type_embedding = nn.Parameter(torch.randn(self.node_embedding_dimension))
        """
        # Load classes weights
        with open(os.path.dirname(__file__) + f"/../../data/classes_weights/{self.config_dataset['experiment_name']}_atoms_weights.json") as file:
            atoms_weights = torch.tensor(list(json.load(file).values()), dtype=torch.float32)
        with open(os.path.dirname(__file__) + f"/../../data/classes_weights/{self.config_dataset['experiment_name']}_bonds_weights.json") as file:
            bonds_weights = torch.tensor(list(json.load(file).values()), dtype=torch.float32)     
        """
        # Loss for single-label, multi-class classification
        self.criterion_atoms = nn.CrossEntropyLoss()#weight = atoms_weights)
        self.criterion_bonds = nn.CrossEntropyLoss()#weight=torch.tensor([1., 1., 1., 1., 1.])) #weight = bonds_weights) 
        
        self.loss_atoms_weight = 1
        # A random predictor has an atom loss 3 times greater than its bond loss.
        # In average, during inference, a molecule has 3 times more bonds predictions than atoms predictions. (counting decoy proposals)
        self.loss_bonds_weight = 3 


    def predict(self, batch):
        predicted_graphs = []
        predictions = self(batch)

        predictions_atoms = [predicted_atom for predicted_atom in predictions[0].detach().cpu().tolist()]
        predictions_bonds = [predicted_bond for predicted_bond in predictions[1].detach().cpu().tolist()]
      
        softmax = nn.Softmax(0)
        confidences = []
        offset_bonds = 0
        offset_atoms = 0
        for batch_index in range(batch["data"].num_graphs):
            data_sub = batch["data"].get_example(batch_index)

            nb_atoms_sub = len([1 for type in data_sub.nodes_types if type == 1])
            nb_bonds_sub = len([1 for type in data_sub.nodes_types if type == 0])

            predictions_atoms_sub = [np.argmax(prediction_atom) for prediction_atom in predictions_atoms[offset_atoms:offset_atoms+nb_atoms_sub]]
            predictions_bonds_sub = [np.argmax(prediction_bond) for prediction_bond in predictions_bonds[offset_bonds:offset_bonds+nb_bonds_sub]]
            confidences_atoms_sub = [torch.max(softmax(torch.tensor(prediction_atom, requires_grad=False))).item() for prediction_atom in predictions_atoms[offset_atoms:offset_atoms+nb_atoms_sub]]
            confidences_bonds_sub = [torch.max(softmax(torch.tensor(prediction_bond, requires_grad=False))).item() for prediction_bond in predictions_bonds[offset_bonds:offset_bonds+nb_bonds_sub]]
            confidences.append(round(min(confidences_atoms_sub + confidences_bonds_sub), 3))

            offset_atoms += nb_atoms_sub
            offset_bonds += nb_bonds_sub

            molecular_graph = MolecularGraph(
                self.config_dataset
            ).from_predictions_nodes_only(
                predictions_atoms_sub, 
                predictions_bonds_sub, 
                data_sub,
                remove_none_predictions = True
            )
            predicted_graphs.append(molecular_graph)

        return predicted_graphs, confidences
    
        
    def forward(self, batch):
        feature_maps = self.backbone(batch["images"])["layer4"] 

        if self.positional_embedding:
            feature_maps = feature_maps + self.node_positional_embedding_map.repeat(self.config_dataset["batch_size"], 1, 1, 1)

        nodes_embeddings_init = torch.zeros([batch["data"].num_nodes, self.node_embedding_dimension], device=self.device)
        i = 0
        
        for data_index in range(batch["data"].num_graphs):
            data_sub = batch["data"].get_example(data_index)

            # Define nodes to atoms and bonds samples mapping
            map_node_sample_index = {}
            sample_index = 0
            for node_index, node_type in enumerate(data_sub.nodes_types):
                if node_type == 1:
                    map_node_sample_index[node_index] = sample_index
                    sample_index += 1
                if node_type == 0:
                    map_node_sample_index[node_index] = list(range(sample_index, sample_index + self.bond_nb_sampled_points))
                    sample_index += self.bond_nb_sampled_points
            nb_samples = sample_index
            
            # Get bonds neighboring atoms indices
            neighboring_atoms_indices = defaultdict(list)
            for index_b, index_e in data_sub.edge_index.t().tolist():
                if data_sub.nodes_types[index_b] == 0: 
                    if index_e not in neighboring_atoms_indices[index_b]:
                        neighboring_atoms_indices[index_b].append(index_e)
                if data_sub.nodes_types[index_e] == 0: 
                    if index_b not in neighboring_atoms_indices[index_e]:
                        neighboring_atoms_indices[index_e].append(index_b)

            # Sample visual nodes embeddings from feature maps
            feature_maps_sub = feature_maps[data_index, :, :, :].unsqueeze(0)
            nodes_grid_sub = torch.zeros(1, nb_samples, 1, 2, device=self.device) 

            for node_index, (x, y) in enumerate(data_sub.nodes_positions):
                if data_sub.nodes_types[node_index] == 1:
                    nodes_grid_sub[0, map_node_sample_index[node_index], 0, 0] = 2*(x/(self.config_dataset["image_size"][1] - 1)) - 1
                    nodes_grid_sub[0, map_node_sample_index[node_index], 0, 1] = -(2*(((self.config_dataset["image_size"][2] - 1) - y)/(self.config_dataset["image_size"][2] - 1)) - 1)
                
                if data_sub.nodes_types[node_index] == 0:
                    bond_positions = [
                        data_sub.nodes_positions[neighboring_atoms_indices[node_index][0]], 
                        data_sub.nodes_positions[neighboring_atoms_indices[node_index][1]]
                    ]

                    # Define bond sampling positions
                    bond_sampling_positions = [
                        get_point_of_trisection_coordinates(bond_positions, ratio_b, ratio_e) 
                            for ratio_b, ratio_e in zip(
                                                        range(1, self.bond_nb_sampled_points + 1), 
                                                        range(self.bond_nb_sampled_points, 0, -1)
                                                    )
                    ]

                    for (x_bond_sample, y_bond_sample), sample_index in zip(bond_sampling_positions, map_node_sample_index[node_index]):
                        # Convert euclidian coordinate system to grid sample coordinate system
                        nodes_grid_sub[0, sample_index, 0, 0] = 2*(x_bond_sample/(self.config_dataset["image_size"][1] - 1)) - 1
                        nodes_grid_sub[0, sample_index, 0, 1] = -(2*(((self.config_dataset["image_size"][2] - 1) - y_bond_sample)/(self.config_dataset["image_size"][2] - 1)) - 1)
                    
            sampled_image_features = F.grid_sample(
                feature_maps_sub, 
                nodes_grid_sub, 
                mode = 'bilinear', 
                align_corners = False,
            )[0, :, :, 0].permute(1, 0)

            # Merge bond sampled features
            bonds_image_features = []
            for node_index in range(data_sub.num_nodes):
                # Store bonds features in the order of nodes indices
                if data_sub.nodes_types[node_index] == 0:
                    sample_indices = map_node_sample_index[node_index]
                    bonds_image_features.append(sampled_image_features[sample_indices])
            
            if len(bonds_image_features) > 0:
                if self.bond_fusion:
                    bonds_image_features = torch.stack(bonds_image_features).permute(0, 2, 1)
                    bonds_image_features = self.bond_fusor(bonds_image_features)[:, :, 0]
                    #print(f"Bond fusor weights: {sum(self.bond_fusor[0].weight)}") # Debug. Check influence of each sampled point
                else:
                    bonds_image_features = torch.mean(torch.stack(bonds_image_features), 1)

            # Set nodes embeddings
            nodes_embeddings_init_sub = []
            bond_index = 0
            for node_index in range(data_sub.num_nodes):
                if data_sub.nodes_types[node_index] == 1:
                    sample_index =  map_node_sample_index[node_index]
                    if self.type_embedding:
                        node_embedding_init = sampled_image_features[sample_index] + self.atom_type_embedding
                    else:
                        node_embedding_init = sampled_image_features[sample_index]
                    
                if data_sub.nodes_types[node_index] == 0:      
                    # Bonds features are read in the node order again
                    if self.type_embedding:
                        node_embedding_init = bonds_image_features[bond_index] + self.bond_type_embedding  
                    else:
                        node_embedding_init = bonds_image_features[bond_index]  

                    bond_index += 1

                nodes_embeddings_init_sub.append(node_embedding_init) 

            nodes_embeddings_init_sub = torch.stack(nodes_embeddings_init_sub)
            
            nodes_embeddings_init[i:i+data_sub.num_nodes, :] = nodes_embeddings_init_sub
            i += data_sub.num_nodes

        # Initialize graphs embeddings
        batch["data"].x = nodes_embeddings_init 
        
        # Classify edges and nodes
        predictions = self.gnn(batch["data"])

        return predictions

    def predict_step(self, batch, batch_idx, dataloader_idx):
        torch.set_num_threads(self.config_dataset["num_threads_pytorch"])
        if len(batch["images"]) < self.config_dataset["batch_size"]:
            print("The proposed batch is smaller than the configurated batch size")
            return None
        return self.predict(batch)

    def training_step(self, batch, batch_idx, dataloader_idx = None):
        predictions = self.forward(batch)

        atoms_indices = [index for index in range(len(batch["data"].nodes_types)) if (batch["data"].nodes_types[index] == 1)]
        bonds_indices = [index for index in range(len(batch["data"].nodes_types)) if (batch["data"].nodes_types[index] == 0)]

        loss_atoms = self.criterion_atoms(predictions[0].float(), batch["data"].nodes_classes[atoms_indices])
        loss_bonds = self.criterion_bonds(predictions[1].float(), batch["data"].nodes_classes[bonds_indices])
        loss = self.loss_atoms_weight*loss_atoms + self.loss_bonds_weight*loss_bonds
        
        # Logging
        self.log("TLA", loss_atoms, prog_bar=False, sync_dist=True, batch_size=self.config_dataset["batch_size"], add_dataloader_idx=False)
        self.log("TLB", loss_bonds, prog_bar=False, sync_dist=True, batch_size=self.config_dataset["batch_size"], add_dataloader_idx=False)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=self.config_dataset["batch_size"], add_dataloader_idx=False)

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx = None):
        predictions = self.forward(batch)

        atoms_indices = [index for index in range(len(batch["data"].nodes_types)) if (batch["data"].nodes_types[index] == 1)]
        bonds_indices = [index for index in range(len(batch["data"].nodes_types)) if (batch["data"].nodes_types[index] == 0)]

        loss_atoms = self.criterion_atoms(predictions[0].float(), batch["data"].nodes_classes[atoms_indices])
        loss_bonds = self.criterion_bonds(predictions[1].float(), batch["data"].nodes_classes[bonds_indices])
        # The loss is a "Molecule-level" metric
        loss = self.loss_atoms_weight*loss_atoms + self.loss_bonds_weight*loss_bonds 
        
        # Compute Node Accuracy. It is a "Molecule-level" metric which gives the proportion of correctly predicted nodes, like the validation loss.
        correct_predictions = 0
        correct_predictions_atoms = 0
        correct_predictions_bonds = 0
        for prediction_atom, gt_atom in zip(predictions[0].float(), batch["data"].nodes_classes[atoms_indices]):
            if torch.argmax(prediction_atom).item() == gt_atom:
                correct_predictions += 1
                correct_predictions_atoms += 1
        for prediction_bond, gt_bond in zip(predictions[1].float(), batch["data"].nodes_classes[bonds_indices]):
            if torch.argmax(prediction_bond).item() == gt_bond:
                correct_predictions += 1
                correct_predictions_bonds += 1
        nb_predictions_atoms = predictions[0].float().size(0)
        nb_predictions_bonds = predictions[1].float().size(0)
        nb_predictions = predictions[0].float().size(0) + predictions[1].float().size(0)
        
        correct = float(int(correct_predictions) == int(nb_predictions))
        accuracy = correct_predictions/nb_predictions * 100
        accuracy_atoms = correct_predictions_atoms/nb_predictions_atoms * 100
        accuracy_bonds = correct_predictions_bonds/nb_predictions_bonds * 100

        # Logging
        if (dataloader_idx == None) or (dataloader_idx == 0):
            suffix = "S"
        else:
            suffix = f"R{dataloader_idx}"

        # In the validation step, the logging is done for each batch
        self.log(f"NAA-{suffix}", accuracy_atoms,  prog_bar=False, sync_dist=True, batch_size=self.config_dataset["batch_size"], add_dataloader_idx=False)
        self.log(f"NAB-{suffix}", accuracy_bonds,  prog_bar=False, sync_dist=True, batch_size=self.config_dataset["batch_size"], add_dataloader_idx=False)
        self.log(f"NA-{suffix}", accuracy,  prog_bar=True, sync_dist=True, batch_size=self.config_dataset["batch_size"], add_dataloader_idx=False)  
        self.log(f"MA-{suffix}", correct,  prog_bar=True, sync_dist=True, batch_size=self.config_dataset["batch_size"], add_dataloader_idx=False)   
        self.log(f"val_loss-{suffix}", loss, prog_bar=True, sync_dist=True, batch_size=self.config_dataset["batch_size"], add_dataloader_idx=False)

        return loss
    
    def configure_optimizers(self):
        #optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.config_training["lr"])
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config_training["lr"])
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

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        if scheduler.optimizer.param_groups[0]["lr"] > 1e-7:
            self.scheduler_step += 1
        scheduler.step(epoch = self.scheduler_step)

    def on_validation_epoch_end(self):     
        """
        End-to-end validation
        """
        if self.trainer.global_rank == 0:   
            if (self.e2e_validation_step != 0) and (self.e2e_validation_step % 5 == 0):
                print("Starting end-to-end validation")
                # Get last checkpoint
                graph_classifier_model_name = max(glob.glob(os.path.dirname(__file__) + f"/../../data/models/graph_classifier/*"), key=os.path.getctime) 
                evaluation_save_path = os.path.dirname(__file__) + f"/../../data/scores/graph_classifier/{graph_classifier_model_name.split('/')[-1][:-5]}_{self.trainer.current_epoch}.json"
                process = subprocess.Popen(
                    f"python3 ../evaluate/evaluate_molecular_recognition.py \
                        --graph-classifier-model-name {graph_classifier_model_name} \
                        --evaluation-save-path {evaluation_save_path} \
                        --benchmarks uspto jpo uob clef uspto-10k uspto-10k-abb uspto-10k-l\
                        --nb-sample-benchmarks 200 200 200 200 200 200 200",
                    shell=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                # Lock until the subprocess terminates
                try:
                    outs, errors = process.communicate(timeout=7200)
                except subprocess.TimeoutExpired:
                   pass
                if (errors != b"") or (outs != b""):
                   pass
                
                with open(evaluation_save_path, 'r') as file:
                    pprint(json.load(file))

            self.e2e_validation_step += 1
