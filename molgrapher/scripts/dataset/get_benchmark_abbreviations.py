#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from mol_depict.molfile_parser.label_molfile import LabelMolFile
from rdkit import Chem
from collections import defaultdict
from pprint import pprint
import json

benchmark = "uspto-10k-abb"
abbreviations_smiles = defaultdict(list)
smiles_abbreviations = defaultdict(list)
for molfile in glob.glob(os.path.dirname(__file__) + f"/../../../data/benchmarks/{benchmark}/molfiles/*"):
    molfile_annotator = LabelMolFile(molfile, reduce_abbreviations=False)
    
    if molfile_annotator.how_many_structures() > 1:
        continue 

    if molfile_annotator.rdk_mol and molfile_annotator.mol and molfile_annotator.mol["sgroups"]:
        sgroups = molfile_annotator.mol["sgroups"]
        
        for sgroup in sgroups:
            sgroup_indices = sgroup["sal_atoms_list"]
            
            attachments = []
            for sgroup_index in sgroup_indices:
                for bond in molfile_annotator.rdk_mol.GetAtomWithIdx(sgroup_index).GetBonds():
                    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                        continue

                    for neighbor_index in [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]:
                        if neighbor_index not in sgroup_indices:
                            attachments.append(neighbor_index)
            
            if len(attachments) == 0:
                print(f"{sgroup} is not attached with single bonds")
                continue

            #if len(attachments) != nb_attachment_point:
            #    continue

            # Mark attachment atom 
            for attachment in attachments:
                molfile_annotator.rdk_mol.GetAtomWithIdx(attachment).SetAtomicNum(0)

            molecule_editable = Chem.EditableMol(molfile_annotator.rdk_mol)
            for index in range(molfile_annotator.rdk_mol.GetNumAtoms()-1, -1, -1):
                if (index not in sgroup_indices) and (index not in attachments):
                    molecule_editable.RemoveAtom(index)

            sgroup_molecule = molecule_editable.GetMol()
            sgroup_smiles = Chem.MolToSmiles(sgroup_molecule)

            # Clean MolFile sgroup label
            label = sgroup["smt_label"].replace("^", "")
            label = label.replace("\\s", "")
            label = label.replace("\\n", "")
            label = label.replace("\\S", "")
            label = label.replace("\\r", "")
            label = label.replace("\r", "")

            # Filter inconsistent (image-molfile) sgroup
            if label in ["O(H)"]: 
                continue
            if ":" in label:
                continue

            # Filer residual markush structures
            if ("~" in label) or (label == "") or (label == "a") or (label == "s") or (label == "q") or (label == "g") or \
                    (label == "e") or (label == "h") or (label == "j") or (label == "i") or (label == "I") or (label == "j2") or \
                    any([ord(e) == 65533 for e in label]):
                continue

            if ("." in sgroup_smiles) or all([c in ["*", "(", ")"] for c in sgroup_smiles]):
                continue
            
            """
            # Debugging
            if label == "(N)":
                print(f"{label} found: ")
                print(molfile, label)
                print("\n")
            """ 
            
            abbreviations_smiles[label].append(sgroup_smiles)
            smiles_abbreviations[sgroup_smiles].append(label)
            #Chem.rdmolfiles.MolToMolFile(sgroup_molecule, "test.mol")

abbreviations_population = {k: len(v) for k, v in abbreviations_smiles.items()}
abbreviations_population_sorted = dict(sorted(abbreviations_population.items(), key=lambda item: item[1]))

abbreviations_smiles_unique = {
    k: {
        "population": abbreviations_population_sorted[k], 
        "smiles": [max(set(v), key=v.count)]
    } 
    for k, v in abbreviations_smiles.items()
}
abbreviations_smiles_unique_sorted = dict(sorted(abbreviations_smiles_unique.items(), key=lambda item: item[1]["population"], reverse=True))
pprint(abbreviations_smiles_unique_sorted)

with open(f"{benchmark}_abbreviations.json", "w") as outfile:
    json.dump(abbreviations_smiles_unique_sorted, outfile)