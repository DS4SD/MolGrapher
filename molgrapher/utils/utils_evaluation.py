#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdkit import Chem, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D
import Levenshtein 
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from collections import defaultdict
from SmilesPE.pretokenizer import atomwise_tokenizer
import json
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from rdkit.Chem.inchi import MolToInchi
import torchvision.ops.boxes as bops
import torch
import math 

from mol_depict.molfile_parser.label_molfile import LabelMolFile
from mol_depict.utils.utils_molecule import get_molecule_from_smiles, get_molecule_from_molfile

def get_molecule_information(smiles=None, molfile_filename=None):
    '''
    - presence of stereochemistry (True, False)
    - number of atoms 
    - largest connectivity of cycles (2 ... n) 
    - largest macrocycle (7 ...) 
    - presence of polyhedral cycles (True, False) 
    - list of exotic heteroatoms (Si, Tn, ...) 
    - presence of common abbreviations (COOH, ...)
    - presence of exotic abbreviations (alk, R-groups...)
    '''
    molecule_information = {
        "stereochemistry": False,
        "nb_atoms": 0,
        #"largest_ring_connectivity": 0,
        "largest_ring_size": 0,
        "ring_mesh": 0,
        "polyhedral_ring": False,
        "exotic_heteroatoms": False,
        #"common_abbreviations": False, 
        #"exotic_abbreviations": False
        "abbreviations": False 
    }

    if molfile_filename:
        molecule = get_molecule_from_molfile(molfile_filename, remove_stereochemistry=False)
        if molecule:
            smiles = Chem.MolToSmiles(molecule)
    else:
        molecule = get_molecule_from_smiles(smiles, remove_stereochemistry=False)
    
    if (molecule == None) or (smiles == ""):
        return molecule_information
    
    # Stereochemistry
    if ("@" in smiles) or ("@@" in smiles):
        molecule_information["stereochemistry"] = True

    # Number of atoms
    molecule_information["nb_atoms"] = molecule.GetNumAtoms()

    # Largest ring size
    rings = molecule.GetRingInfo()
    atom_rings = rings.AtomRings()
    if len(atom_rings) != 0:
        largest_ring = max(atom_rings, key = len)
        molecule_information["largest_ring_size"] = len(largest_ring)

    """
    # Ring mesh
    if len(atom_rings) != 0:
        ring_connections = {id: [] for id in range(len(atom_rings))}
        #connected_rings = []
        for ring_id in range(len(atom_rings)):
            for other_ring_id in range(len(atom_rings)):
                if other_ring_id != ring_id and len(list(set(atom_rings[ring_id]).intersection(atom_rings[other_ring_id])))>=2:
                    ring_connections[ring_id].append(other_ring_id)

        edges = get_edges(ring_connections)
        G = defaultdict(list)
        for (s,t) in edges:
            G[s].append(t)
            G[t].append(s)
        G = list(G.values())
        print(G)
        all_paths = [p for ps in [dfs(G, n) for n in set(G)] for p in ps]
        if len(all_paths) > 1:
            max_len = max(len(p) for p in all_paths)
            molecule_information["ring_mesh"] = max_len
    """
    # Polyhedral ring (make a list of possibilities and check presence)
    '''
    For rings of 5 atoms, at least 3 atoms must be shared with another ring.
    For rings with 6 or more atoms, at least 4.
    To be improved: what for rings <5 atoms?
                    How to get rid of macrocycles that have intrinsic rings? Limit so the polyhedral occupies half of the ring at least.
                    Should we get rid of those at all? YES
    For now, focus only on the next three examples:
    C1CC2CCC1CC2-C1CC2CC1CC2-C1C2CC3CC1CC(C2)C3 + derivates without stereochemistry
    '''
    polyhedral_molecules = [
        "C1CC2CCC1CC2","C1CC2CCC(C1)CC2","C1=CC2C=CC1CC2","C1CC2NCC1CN2","C1CN2CCC1CC2","C1CC2CCC1CNC2",
        "C1CC2CCC(CC2)N1","C1CC2CC1C2","C1C2CC3C1C3C2","C1=CC2C=CC1C2","C1OC2CC1CO2","C1C[C@H]2CNC[C@@H]1C2","C1=CC2C=CC1C2",
        "C1CC2CC3CC1CC(C2)C3","C1C2CC3CC1CC(C2)S3","C1C2CC3CC1CC(C2)O3","C1C2CC3CC1CC(C2)N3","C1C2CC3CC1CC(O2)O3","N1CC2CC(C1)CNC2",
        "C1C2CC3CC1CN(C2)C3","C1C2CN3CN1CN(C2)C3","C1B2CC3CC1CC(C2)C3","C1NC2CC3CC1CC(C2)C3","C1CC2CC1CC2","C1C2CC3CC1CC(C2)C3",
        "C1CC2CCCC1CNC2","C1CC2CCCC1CCC2"
    ]
    for poly_smiles in polyhedral_molecules:
        poly_mol = Chem.MolFromSmiles(poly_smiles)
        if len(molecule.GetSubstructMatches(poly_mol)) != 0:
            molecule_information["polyhedral_ring"] = True
            break

    # Exotic heteroatoms (Common: O, N, Cl, I, P, Li, Mg, Br, S)
    exotic_heteroatoms = [
        "He", "Be", "B", "Ne", "Na", "Al",
        "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
        "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "Xe", "Cs", "Ba", "La", "Ce",
        "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
        "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
        "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Uut",
        "Fl", "Uup", "Lv", "Uus", "Uuo"
    ]
    molecule_atoms = atomwise_tokenizer(smiles)
    molecule_atoms = [atom.replace('[','').replace(']','').replace('H','').replace('@','') for atom in molecule_atoms]
    for exotic_hetam in exotic_heteroatoms:
        if exotic_hetam in molecule_atoms:
            molecule_information["exotic_heteroatoms"] = True
            break

    # Common abbreviations
    common_abbreviations = ["Met", "COOH", "R", "", "", "", "", ""]

    # Exotic abbreviations
    exotic_abbreviations = ["alk", ] 

    # Abbreviations
    #if molfile_filename is not False:
    #    molfile_annotator = LabelMolFile(molfile_filename)
    #    if molfile_annotator.are_abbreviated()[0] == True:
    #        molecule_information["abbreviations"] = True

    return molecule_information

def get_edges(graph):
    edges = []
    for node in graph:
        for neighbour in graph[node]:
            edges.append((node, neighbour))
    return edges

def dfs(G, start):
    """Perform dfs on adjacency list of graph
    G: Adjacency list representation of graph.
    start: Index of start node.
    """
    visited = [False] * len(G)
    stack = []
    path = []
    stack.append(start)
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            path.append(v)
            for neighbor in G[v]:
                stack.append(neighbor)
    return(path)

def compute_molecule_prediction_quality(predicted_smiles, gt_smiles, predicted_molecule=None, gt_molecule=None, remove_stereo=True, remove_double_bond_stereo=True):
    scores = {"levenshtein": len(gt_smiles), "levenshtein0": False,
              "tanimoto": 0, "tanimoto1": False, 
              "bleu_average": 0, "bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0,
              "rouge1": 0, "rouge2": 0, "rouge3": 0, "rouge4": 0, "rougeL": 0,
              "valid": False, "correct": False}

    if predicted_smiles is None or (isinstance(predicted_smiles, float) and math.isnan(predicted_smiles)):
        return scores

    if Chem.MolFromSmiles(predicted_smiles) is None:
        return scores

    # Levenshtein distance
    levenshtein = Levenshtein.distance(predicted_smiles, gt_smiles)
    scores["levenshtein"] = levenshtein
    if levenshtein == 0:
        scores["levenshtein0"] = True
    
    # Tanimoto score 
    if not predicted_molecule:
        predicted_molecule = get_molecule_from_smiles(predicted_smiles, remove_stereochemistry = remove_stereo)
    if not gt_molecule:
        gt_molecule = get_molecule_from_smiles(gt_smiles, remove_stereochemistry = remove_stereo)
    if remove_double_bond_stereo:
        for bond in gt_molecule.GetBonds():
            bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
        for bond in predicted_molecule.GetBonds():
            bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)

    # Remove aromatic bonds (Kekulize)
    try:
        gt_molecule = rdMolDraw2D.PrepareMolForDrawing(gt_molecule, addChiralHs=False)
    except:
        print(f"{gt_smiles} can't be kekulized")
        return scores
    predicted_molecule = rdMolDraw2D.PrepareMolForDrawing(predicted_molecule, addChiralHs=False)
    
    scores["tanimoto"] = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_molecule), Chem.RDKFingerprint(predicted_molecule))
    scores["tanimoto1"] = scores["tanimoto"] == 1
    
    # Inchi equality
    if remove_stereo:
        if MolToInchi(predicted_molecule, options="/SNon") == MolToInchi(gt_molecule, options="/SNon"):
            scores["correct"] = True
    else:
        if MolToInchi(predicted_molecule) == MolToInchi(gt_molecule):
            scores["correct"] = True

    # BLEU score
    scores["bleu_average"] = sentence_bleu([[c for c in gt_smiles]], [c for c in predicted_smiles], weights=[0.25, 0.25, 0.25, 0.25], smoothing_function=SmoothingFunction().method1)
    scores["bleu1"] = sentence_bleu([[c for c in gt_smiles]], [c for c in predicted_smiles], weights=[1, 0, 0, 0], smoothing_function=SmoothingFunction().method1)
    scores["bleu2"] = sentence_bleu([[c for c in gt_smiles]], [c for c in predicted_smiles], weights=[0, 1, 0, 0], smoothing_function=SmoothingFunction().method1)
    scores["bleu3"] = sentence_bleu([[c for c in gt_smiles]], [c for c in predicted_smiles], weights=[0, 0, 1, 0], smoothing_function=SmoothingFunction().method1)
    scores["bleu4"] = sentence_bleu([[c for c in gt_smiles]], [c for c in predicted_smiles], weights=[0, 0, 0, 1], smoothing_function=SmoothingFunction().method1)

    # ROUGE score
    scores_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL'], use_stemmer=True).score(" ".join([c for c in gt_smiles]), " ".join([c for c in predicted_smiles]))
    scores["rouge1"] = scores_rouge['rouge1'].fmeasure
    scores["rouge2"] = scores_rouge['rouge2'].fmeasure
    scores["rouge3"] = scores_rouge['rouge3'].fmeasure
    scores["rouge4"] = scores_rouge['rouge4'].fmeasure
    scores["rougeL"] = scores_rouge['rougeL'].fmeasure

    # Valid SMILES
    scores["valid"] = True
    
    return scores

def flatten_list(input_list):
    return [item for sublist in input_list for item in sublist]

def get_metrics(predictions, selection):
    vocabulary_atoms = json.load(open(os.getcwd() + f"/../../data/vocabularies/vocabulary_atoms_{config['nb_atoms_classes']}.json"))
    vocabulary_bonds = json.load(open(os.getcwd() + f"/../../data/vocabularies/vocabulary_atoms_{config['nb_bonds_classes']}.json"))

    gt_atoms = [prediction for index, prediction in enumerate(predictions["gt_nodes"]) if index in selection]
    predicted_atoms = [prediction for index, prediction in enumerate(predictions["predictions_nodes"]) if index in selection]
    gt_bonds = [prediction for index, prediction in enumerate(predictions["gt_edges"]) if index in selection]
    predicted_bonds = [prediction for index, prediction in enumerate(predictions["predictions_edges"]) if index in selection]

    gt_atoms_flat = flatten_list(gt_atoms)
    predicted_atoms_flat = flatten_list(predicted_atoms)
    gt_bonds_flat = flatten_list(gt_bonds)
    predicted_bonds_flat = flatten_list(predicted_bonds)

    binarizer = MultiLabelBinarizer().fit(predicted_atoms + gt_atoms)
 
    metrics = {
        "atoms": {
            "confusion_matrix": confusion_matrix(gt_atoms_flat, predicted_atoms_flat, labels=list(vocabulary_atoms.values())).tolist(),
            "classification_report": classification_report(gt_atoms_flat, predicted_atoms_flat, output_dict=True, target_names=list(vocabulary_atoms.keys()), labels=list(vocabulary_atoms.values())),
        },
        "bonds": {
            "confusion_matrix": confusion_matrix(gt_bonds_flat, predicted_bonds_flat, labels=list(vocabulary_atoms.values())).tolist(),
            "classification_report": classification_report(gt_bonds_flat, predicted_bonds_flat, output_dict=True, target_names=list(vocabulary_bonds.keys()), labels=list(vocabulary_bonds.values())),
        },
        "molecules": {
            "atoms": accuracy_score(binarizer.transform(gt_atoms), binarizer.transform(predicted_atoms)),
            "bonds": accuracy_score(binarizer.transform(gt_bonds), binarizer.transform(predicted_bonds))
        }
    }

    return metrics

def get_metrics_keypoint_detector(gt_keypoints, predicted_keypoints, test_window = 10):
    gt_keypoints = torch.tensor([[
        gt_keypoint[0] - test_window, 
        gt_keypoint[1] - test_window, 
        gt_keypoint[0] + test_window, 
        gt_keypoint[1] + test_window
    ] for gt_keypoint in gt_keypoints], dtype=torch.float)

    predicted_keypoints = torch.tensor([[
        predicted_keypoint[0] - test_window, 
        predicted_keypoint[1] - test_window, 
        predicted_keypoint[0] + test_window, 
        predicted_keypoint[1] + test_window
    ] for predicted_keypoint in predicted_keypoints], dtype=torch.float)

    for gt_i, gt_keypoint in enumerate(gt_keypoints):
        others_gt_keypoints = gt_keypoints[[i for i in range(len(gt_keypoints)) if (i!= gt_i)]]
        if bops.box_iou(gt_keypoint.unsqueeze(0), others_gt_keypoints).sum() > 0:
            print("ERROR in get_metrics_keypoint_detector: Some of the ground truth keypoints are too close.")
            return None, None
    tp = 0
    for gt_keypoint in gt_keypoints:
        for predicted_keypoint in predicted_keypoints:
            if bops.box_iou(predicted_keypoint.unsqueeze(0), gt_keypoint.unsqueeze(0)).item() > 0:
                tp +=1
                break
                
    precision = tp/len(predicted_keypoints)
    recall = tp/len(gt_keypoints)
    return precision, recall
