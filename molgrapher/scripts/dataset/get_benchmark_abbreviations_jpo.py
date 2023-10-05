#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from mol_depict.molfile_parser.label_molfile import LabelMolFile
from rdkit import Chem
from collections import defaultdict
from pprint import pprint
import numpy as np
from mol_depict.utils.utils_drawing import draw_molecule_rdkit
import json
from rdkit.Chem import rdAbbreviations

abbreviations_population = defaultdict(int)
# Get all abbreviations
for molfile in glob.glob(os.path.dirname(__file__) + "/../../../data/benchmarks/jpo/molfiles/*"):
    if "2008056719_47_chem.sdf" in molfile:
        continue

    sgroups = []
    abbreviated_molecule = False
    with open(molfile, 'rb') as file:
        lines = file.readlines()
        for line in lines:
            line = str(line.strip())
            if ("LABEL" in line) and ("ATOMS" in line):
                abbreviated_molecule = True

    if not abbreviated_molecule:
        continue

    molecule = Chem.MolFromMolFile(molfile)

    try:
        condensed_molecule = rdAbbreviations.CondenseAbbreviationSubstanceGroups(molecule)
    except:
        continue
    
    for atom in condensed_molecule.GetAtoms():
        if atom.GetSymbol() == "*":
            abbreviations_population[atom.GetProp("atomLabel")] += 1

abbreviations_population_sorted = dict(sorted(abbreviations_population.items(), key=lambda item: item[1]))
print(abbreviations_population_sorted)

# Get abbreviations - smiles mapping when it is possible
abbreviations_smiles = defaultdict(list)
smiles_abbreviations = defaultdict(list)
for molfile in glob.glob(os.path.dirname(__file__) + "/../../../data/benchmarks/jpo/molfiles/*"):
    if "2008056719_47_chem.sdf" in molfile:
        continue

    sgroups = []
    abbreviated_molecule = False
    with open(molfile, 'rb') as file:
        lines = file.readlines()
        for line in lines:
            line = str(line.strip())
            if ("LABEL" in line) and ("ATOMS" in line):
                abbreviated_molecule = True

    if not abbreviated_molecule:
        continue

    molecule = Chem.MolFromMolFile(molfile)

    try:
        condensed_molecule = rdAbbreviations.CondenseAbbreviationSubstanceGroups(molecule)
    except:
        continue

    #draw_molecule_rdkit(smiles = "", molecule = molecule, path = "test1.png", augmentations = True, fake_molecule = False)
    #draw_molecule_rdkit(smiles = "", molecule = condensed_molecule, path = "test2.png", augmentations = True, fake_molecule = False)
    # Get abbreviated atoms except the preserved one by checking position overlaps
    atoms_positions = {}
    for i in range(molecule.GetNumAtoms()):
        position = molecule.GetConformer(0).GetAtomPosition(i)
        atoms_positions[i] = [position.x, position.y]

    sgroups = []
    for index_1 in range(len(atoms_positions)):
        for index_2 in range(len(atoms_positions)):
            if index_1 != index_2:
                if atoms_positions[index_1] == atoms_positions[index_2]:
                    matched = False
                    for sgroup in sgroups:
                        if sgroup["position"] == atoms_positions[index_1]:
                            if index_1 not in sgroup["indices"]:
                                sgroup["indices"].append(index_1)
                            if index_2 not in sgroup["indices"]:
                                sgroup["indices"].append(index_2)
                            matched = True
                            break
                    if matched:
                        break
                    new_sgroup = {
                        "position": atoms_positions[index_1],
                        "indices": [index_1, index_2]
                    }
                    
                    sgroups.append(new_sgroup)

    # Add the preserved atom to the abbreviation
    for sgroup in sgroups:
        for sgroup_index in sgroup["indices"]:
            for bond in molecule.GetAtomWithIdx(sgroup_index).GetBonds():
                for neighbor_index in [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]:
                    if neighbor_index not in sgroup["indices"]:
                        try:
                            abbreviated_atom = condensed_molecule.GetAtomWithIdx(neighbor_index)
                        except: 
                            # If the neighboring atom can't be found in the abbreviated molecule, it should added to the abbreviation
                            sgroup["indices"].append(neighbor_index)

                        if (abbreviated_atom.GetSymbol() == "*") and abbreviated_atom.HasProp("atomLabel"):
                            sgroup["smt_label"] = abbreviated_atom.GetProp("atomLabel")
                            sgroup["indices"].append(neighbor_index)

    for sgroup in sgroups:
        attachments = []
        for sgroup_index in sgroup["indices"]:
            for bond in molecule.GetAtomWithIdx(sgroup_index).GetBonds():
                for neighbor_index in [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]:
                    if neighbor_index not in sgroup["indices"]:
                        attachments.append(neighbor_index)

        if len(attachments) != 1:
            continue

        # Mark attachment atom 
        molecule.GetAtomWithIdx(attachments[0]).SetAtomicNum(0)

        molecule_editable = Chem.EditableMol(molecule)
        for index in range(molecule.GetNumAtoms()-1, -1, -1):
            if (index not in sgroup["indices"]) and (index != attachments[0]):
                molecule_editable.RemoveAtom(index)

        sgroup_molecule = molecule_editable.GetMol()
        
        if "smt_label" in sgroup:
            abbreviations_smiles[sgroup["smt_label"]].append(Chem.MolToSmiles(sgroup_molecule))

abbreviations_smiles_unique = {
    k: {
        "population": abbreviations_population_sorted[k], 
        "smiles": list(np.unique(v))
    } 
    for k, v in abbreviations_smiles.items()
}
abbreviations_smiles_unique_sorted = dict(sorted(abbreviations_smiles_unique.items(), key=lambda item: item[1]["population"], reverse=True))
print(abbreviations_smiles_unique_sorted)

with open("jpo_abbreviations.json", "w") as outfile:
    json.dump(abbreviations_smiles_unique_sorted, outfile)
       