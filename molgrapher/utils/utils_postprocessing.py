#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdkit import Chem 
from rdkit.Chem import rdDepictor, rdmolfiles
from rdkit.Geometry.rdGeometry import Point3D
import os 
import math
import numpy as np
from numpy.linalg import norm
import copy
from collections import Counter
from pprint import pprint 
import time 

from mol_depict.utils.utils_drawing import draw_molecule_rdkit

class GraphPostprocessor:
    def __init__(
        self, molecular_graph, abbreviations, abbreviations_smiles_mapping, ocr_atoms_classes_mapping, spelling_corrector, 
        postprocessing_flags=None, align_rdkit_output=False, assign_stereo=False, remove_hydrogens=False, filename_logging="", 
        molecule=None, needs_update_stereo=None, bonds_directions=None, 
        ):
        self.molecular_graph = molecular_graph
        self.molecule = molecule
        self.abbreviations = abbreviations
        self.abbreviations_smiles_mapping = abbreviations_smiles_mapping
        self.ocr_atoms_classes_mapping = ocr_atoms_classes_mapping
        self.spelling_corrector = spelling_corrector
        self.align_rdkit_output = align_rdkit_output
        self.remove_hydrogens = remove_hydrogens
        self.assign_stereo = assign_stereo
        self.needs_update_stereo = needs_update_stereo
        self.bonds_directions = bonds_directions
        self.superatoms = []
        self.filename_logging = filename_logging
        self.original_abbreviations = abbreviations
        # Postprocessing steps flags (to speed up processing or avoid infinite recursions)
        if postprocessing_flags is None: 
            postprocessing_flags = {}
        self.postprocessing_flags = postprocessing_flags
    
    def remove_hydrogens_on_carbons(self, smiles):
        molecule = Chem.MolFromSmiles(smiles, sanitize=False)
        is_sanity = Chem.SanitizeMol(
            molecule,
            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION, 
            catchErrors=True,
        )        
        if molecule is None:
            return smiles
        editable_molecule = Chem.RWMol(molecule)
        chir_centers = set([c[0] for c in Chem.FindMolChiralCenters(molecule)])
        editable_molecule.BeginBatchEdit()
        for match in molecule.GetSubstructMatches(Chem.MolFromSmarts("[#1][#6]")):
            aid = match[0]
            h_atom = molecule.GetAtomWithIdx(aid)
            if h_atom.GetIsotope() > 1:
                continue
            cid = match[1]
            if cid in chir_centers:
                continue
            editable_molecule.RemoveAtom(aid)        
        for match in molecule.GetSubstructMatches(Chem.MolFromSmarts("[#1]=[*]")):
            aid = match[0]
            editable_molecule.RemoveAtom(aid)        
        for match in molecule.GetSubstructMatches(Chem.MolFromSmarts("[#1]#[*]")):
            aid = match[0]
            editable_molecule.RemoveAtom(aid)        
        editable_molecule.CommitBatchEdit()
        return Chem.MolToSmiles(editable_molecule.GetMol())
    
    def postprocess_incorrect_valence_tailored(self, smiles):
        #print("postprocess_incorrect_valence_tailored", smiles)       
        if self.remove_hydrogens:
            smiles = self.remove_hydrogens_on_carbons(smiles)        
            
        # Remove hydrogens connected to S, Te, N
        molecule = Chem.MolFromSmiles(smiles, sanitize=False)
        if molecule is None:
            return None
        if Chem.SanitizeMol(molecule, Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION, catchErrors=True) != Chem.rdmolops.SANITIZE_NONE:
            return None
        
        queries = [
            (Chem.MolFromSmarts("[#16;0;v4D2](=[#6])[#6]"), 1, 0, None),              # [SH](=C)C -> [S+]
            (Chem.MolFromSmarts("[#16;0;v4D3]([#6])([#6])[#6]"), 1, 0, None),         # [SH](C)(C)C -> [S+]
            (Chem.MolFromSmarts("[#16;0;v4D4]([#1])([#6])([#6])[#6]"), 1, 0, 1),      # [S]([H])(C)(C)C -> [S+]
            (Chem.MolFromSmarts("[#16;0;v3D3]([#6])([#6])[#6]"), 1, 0, None),         # [S](C)(C)C -> [S+]
            (Chem.MolFromSmarts("[#16;0;v4D3]([#1])(=[#6])[#6]"), 1, 0, 1),           # [S]([H])(=C)C -> [S+]
            (Chem.MolFromSmarts("[#16;0;v3D2](=[#6])[#6]"), 1, 0, None),              # [S](=C)C -> [S+]
            
            (Chem.MolFromSmarts("[#52;0;v4D2](=[#6])[#6]"), 1, 0, None),              # [TeH](=C)C -> [Te+]
            (Chem.MolFromSmarts("[#52;0;v4D3]([#6])([#6])[#6]"), 1, 0, None),         # [TeH](C)(C)C -> [Te+]
            (Chem.MolFromSmarts("[#52;0;v4D4]([#1])([#6])([#6])[#6]"), 1, 0, 1),      # [Te]([H])(C)(C)C -> [Te+]
            (Chem.MolFromSmarts("[#52;0;v3D3]([#6])([#6])[#6]"), 1, 0, None),         # [Te](C)(C)C -> [Te+]
            (Chem.MolFromSmarts("[#52;0;v4D3]([#1])(=[#6])[#6]"), 1, 0, 1),           # [Te]([H])(=C)C -> [Te+]
            (Chem.MolFromSmarts("[#52;0;v3D2](=[#6])[#6]"), 1, 0, None),              # [Te](=C)C -> [Te+]
            
            (Chem.MolFromSmarts("[#7;0;v4D4]([#6])([#6])([#6])[#6]"), 1, 0, None),    # [N](C)(C)(C)C -> [N+]
            (Chem.MolFromSmarts("[#7;0;v5D5]([#1])([#6])([#6])([#6])[#6]"), 1, 0, 1), # [N]([H])(C)(C)(C)C -> [N+]
            (Chem.MolFromSmarts("[#7;0;v4D3](=[#6])([#6])[#6]"), 1, 0, None),         # [N](=C)(C)C -> [N+]
            (Chem.MolFromSmarts("[#7;0;v5D4]([#1])(=[#6])([#6])[#6]"), 1, 0, 1),      # [N]([H])(=C)(C)C -> [N+]
        ]

        for query, charge, explicit_hydrogen, remove_atom_idx in queries:
            for match in molecule.GetSubstructMatches(query):
                i_atom = match[0]
                atom = molecule.GetAtomWithIdx(i_atom)
                atom.SetFormalCharge(charge)
                if explicit_hydrogen is not None:
                    atom.SetNumExplicitHs(explicit_hydrogen)
                if remove_atom_idx is not None:
                    i_atom = match[remove_atom_idx]
                    editable_molecule = Chem.RWMol(molecule)
                    editable_molecule.BeginBatchEdit()
                    editable_molecule.RemoveAtom(i_atom)
                    editable_molecule.CommitBatchEdit()
                    molecule = editable_molecule.GetMol()
        
        smiles = Chem.MolToSmiles(molecule)
        return smiles

    def postprocess_before_rdkit_molecule_creation(self):
        # Overwrite deuterium atoms (Graph post-processing)
        if ("postprocessed_deuterium" not in self.postprocessing_flags):
            self.overwrite_deuterium_atoms()
        # Overwrite vertical abbreviations SO2 and CF2 (Graph post-processing)
        if ("postprocessed_vertical_abbreviations" not in self.postprocessing_flags):
            self.overwrite_vertical_abbreviation()
        # Overwrite carbon_dioxide (Graph post-processing)
        if ("postprocessed_carbon_dioxide" not in self.postprocessing_flags):
            self.overwrite_carbon_dioxide()
    
    def postprocess_after_rdkit_molecule_creation(self):
        # Post-process aromatic bonds not in cycles (RDKit molecule post-processing)
        self.postprocess_aromatic_bonds()
        # Post-process aromatic rings (RDKit molecule post-processing)
        self.postprocess_aromatic_rings()
        # Post-process polycycles - specific patch (RDKit molecule post-processing)
        self.postprocess_polycycles_tailored()
        # Post-process polycycles (RDKit molecule post-processing)
        self.postprocess_polycycles()
        
        # Define molecule conformation for alignment
        if self.align_rdkit_output:
            molecule_aligned = copy.deepcopy(self.molecule)
            rdDepictor.Compute2DCoords(molecule_aligned) 
            for atom_index, atom in enumerate(self.molecular_graph.atoms):
                position = Point3D()
                position.x, position.y, position.z = atom["position"][0]/64, -atom["position"][1]/64, 0
                molecule_aligned.GetConformer(0).SetAtomPosition(atom_index, position)
                
        # Return the molecule if it does not contain any abbreviations and is sanitized
        if all([(self.molecular_graph.atoms_classes_symbols[atom["class"]] != "R") for atom in self.molecular_graph.atoms]) and len(Chem.GetMolFrags(self.molecule)) == 1:
            #print(f"{self.filename_logging}:Return molecule without abbreviations.")
            if self.align_rdkit_output:
                try:
                    rdDepictor.GenerateDepictionMatching2DStructure(
                        self.molecule,
                        reference = molecule_aligned
                    )
                    recompute_coords = False
                except Exception as e:
                    print(f"{self.filename_logging}:Aligning RDKit molecule failed: {e}")
                    recompute_coords = True
            else:
                recompute_coords = True
            # Assign stereo-chemistry
            if self.assign_stereo and self.needs_update_stereo:
                self.molecule = self.assign_stereo_centers(self.molecule)
                
            molecule_return = self.try_sanitize_molecule(self.molecule, recompute_coords=recompute_coords)
            if molecule_return:
                return molecule_return
        
        original_molecule = copy.deepcopy(self.molecule)
        if self.assign_stereo and self.needs_update_stereo:
            original_molecule = self.assign_stereo_centers(original_molecule)
        self.original_abbreviations = copy.deepcopy(self.abbreviations)
        
        # Resolve abbreviations (RDKit molecule post-processing)
        self.resolve_abbreviations()
        
        # Assign stereo-chemistry (RDKit molecule post-processing)
        if self.assign_stereo and self.needs_update_stereo:
            self.molecule = self.assign_stereo_centers(self.molecule)
        
        # Return the molecule if it is sanitized
        if len(Chem.GetMolFrags(self.molecule)) == 1:
            #print(f"{self.filename_logging}:Return molecule with {sum([(self.molecular_graph.atoms_classes_symbols[atom['class']] == 'R') for atom in self.molecular_graph.atoms])} abbreviations.")
            if self.align_rdkit_output:
                molecule_aligned_2 = Chem.EditableMol(molecule_aligned)
                for abbreviation_index in sorted(self.matched_abbreviations_indices, reverse=True):
                    molecule_aligned_2.RemoveAtom(abbreviation_index)
                molecule_aligned_2 = molecule_aligned_2.GetMol()
                try:
                    rdDepictor.GenerateDepictionMatching2DStructure(
                        self.molecule,
                        reference = molecule_aligned_2
                    )
                    recompute_coords = False
                except Exception as e:
                    print(f"{self.filename_logging}:Aligning RDKit molecule failed: {e}")
                    recompute_coords = True
            else:
                recompute_coords = True
            molecule_return = self.try_sanitize_molecule(self.molecule, recompute_coords=recompute_coords)
            if molecule_return:
                return molecule_return
        
        # Else merge keypoints (Graph post-processing)
        elif ("postprocessed_keypoints" not in self.postprocessing_flags) and (len(self.original_abbreviations) > 0):
            output = self.merge_keypoints()
            if output is not None:
                print(f"{self.filename_logging}:Recursive molecule creation (after merging keypoints using PaddleOCR predicted boxes)")
                return output
        
        # If merging keypoints fails, return the original molecule if it is sanitized
        if len(Chem.GetMolFrags(original_molecule)) == 1:
            print(f"{self.filename_logging}:Merging keypoints failed. Return original molecule.")
            if self.align_rdkit_output:
                molecule_aligned_2 = Chem.EditableMol(molecule_aligned)
                for abbreviation_index in sorted(self.matched_abbreviations_indices, reverse=True):
                    molecule_aligned_2.RemoveAtom(abbreviation_index)
                molecule_aligned_2 = molecule_aligned_2.GetMol()
                try:
                    rdDepictor.GenerateDepictionMatching2DStructure(
                        self.molecule,
                        reference = molecule_aligned_2
                    )
                    recompute_coords = False
                except Exception as e:
                    print(f"{self.filename_logging}:Aligning RDKit molecule failed: {e}")
                    recompute_coords = True
            else:
                recompute_coords = True
            original_molecule_return = self.try_sanitize_molecule(original_molecule, recompute_coords=recompute_coords)
            if original_molecule_return:
                return original_molecule_return
        
        # Remove single atom without connections (Graph post-processing)
        if ("postprocessed_isolated_atoms" not in self.postprocessing_flags) and (len(Chem.GetMolFrags(original_molecule)) > 1):
            output = self.remove_isolated_atoms()
            if output is not None:
                print(f"{self.filename_logging}:Recursive molecule creation (after removing isolated atoms)")
                return output
        
        # todo: return largest fragment, try to connect multiple fragments
        print(f"{self.filename_logging}:Predicted molecule has multiple fragments")
        return Chem.MolFromSmiles("C")
        
    def try_sanitize_molecule(self, molecule, recompute_coords=False):
        try:
            # Fix specific valence issues
            postprocessed_molecule = Chem.MolFromSmiles(self.postprocess_incorrect_valence_tailored(Chem.MolToSmiles(molecule)), sanitize=False)
            if not(postprocessed_molecule is None):
                molecule = postprocessed_molecule
            
            # Try minimal sanitization
            Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            if recompute_coords:
                rdDepictor.Compute2DCoords(molecule) 
                
            return molecule
        except Exception as e:
            print(f"{self.filename_logging}:The predicted molecule can not be sanitized: {e}")
            return False

    def overwrite_deuterium_atoms(self):
        one_deuterium_postprocessed = False
        for i, abbreviation in enumerate(self.abbreviations):
            if abbreviation["text"] != "D":
                continue            
            for abbreviation_index, atom in enumerate(self.molecular_graph.atoms): 
                if (atom["position"][0] >= (abbreviation["box"][0][0])) and \
                    (atom["position"][0] <= (abbreviation["box"][1][0])) and \
                    (atom["position"][1] >= (abbreviation["box"][0][1])) and \
                    (atom["position"][1] <= (abbreviation["box"][1][1])):

                    # Overwrite atoms of classes... 
                    if (atom["class"] != self.molecular_graph.symbols_classes_atoms["O"]) and \
                       (atom["class"] != self.molecular_graph.symbols_classes_atoms["R"]):
                        #(atom["class"] != self.molecular_graph.symbols_classes_atoms["P"]) and \
                        #(atom["class"] != self.molecular_graph.symbols_classes_atoms["B"]):
                        continue 
                    
                    # ... which are connected with one single bond
                    neighboring_bonds = self.molecular_graph.get_neighboring_bonds_from_atom(abbreviation_index)
                    
                    if ("DASHED" in self.molecular_graph.types_classes_bonds) and ("SOLID" in self.molecular_graph.types_classes_bonds):
                        if not((len(neighboring_bonds) == 1) and \
                            ((neighboring_bonds[0]["class"] == self.molecular_graph.types_classes_bonds["SINGLE"]) or \
                             (neighboring_bonds[0]["class"] == self.molecular_graph.types_classes_bonds["DASHED"]) or
                             (neighboring_bonds[0]["class"] == self.molecular_graph.types_classes_bonds["SOLID"]))):
                            continue
                    else:
                        if not((len(neighboring_bonds) == 1) and \
                            (neighboring_bonds[0]["class"] == self.molecular_graph.types_classes_bonds["SINGLE"])):
                            continue
                    
                    # Convert atom to abbreviation node
                    self.molecular_graph.atoms[abbreviation_index]["class"] = self.molecular_graph.symbols_classes_atoms["R"]
                    one_deuterium_postprocessed = True
        self.postprocessing_flags["postprocessed_deuterium"] = True
      
    def overwrite_carbon_dioxide(self):
        """Removes carbon atom predicted at the charge location"""   
        one_carbon_dioxide_postprocessed = False
        for abbreviation_index, atom in enumerate(self.molecular_graph.atoms): 
            if not(self.molecular_graph.atoms_classes_symbols[atom["class"]] == "R"):
                continue
            for i, abbreviation in enumerate(self.abbreviations):
                if not((atom["position"][0] >= (abbreviation["box"][0][0])) and \
                    (atom["position"][0] <= (abbreviation["box"][1][0])) and \
                    (atom["position"][1] >= (abbreviation["box"][0][1])) and \
                    (atom["position"][1] <= (abbreviation["box"][1][1]))):
                    continue
                charge_removed = False 
                if ((abbreviation["text"] != "CO2") and (abbreviation["text"] != "CO2-")):
                    # Maybe "CO2" should be removed.
                    continue
                
                for connection_atom in self.molecular_graph.atoms:
                    if (len(self.molecular_graph.get_neighbors(connection_atom)) != 1) or \
                        (connection_atom["class"] != self.molecular_graph.symbols_classes_atoms["C"]):
                        continue
                    print(f"{self.filename_logging}:Try to post-process CO2 ocr predictions if the number of connection points mismatch")
                    # Remove bonds
                    remove_bonds_indices = []
                    atoms_matched_indices = [connection_atom["index"]]
                    for bond_index, bond in enumerate(self.molecular_graph.bonds):
                        if (bond["index"][0] in atoms_matched_indices) or (bond["index"][1] in atoms_matched_indices):
                            remove_bonds_indices.append(bond_index)
                    self.molecular_graph.bonds = [bond for bond_index, bond in enumerate(self.molecular_graph.bonds) if bond_index not in remove_bonds_indices]

                    # Remove atoms
                    self.molecular_graph.atoms = [atom for atom_index, atom in enumerate(self.molecular_graph.atoms) if atom_index not in atoms_matched_indices]
                    
                    # Update bond indices
                    for bond in self.molecular_graph.bonds:
                        b = bond["index"][0]
                        e = bond["index"][1]
                        b -= sum([b > removed_atom_index for removed_atom_index in atoms_matched_indices])
                        e -= sum([e > removed_atom_index for removed_atom_index in atoms_matched_indices])
                        bond["index"] = [b, e]
                    charge_removed = True
                    one_carbon_dioxide_postprocessed = True
                    break
                if charge_removed:
                    break
        self.postprocessing_flags["postprocessed_carbon_dioxide"] = True
       
    def overwrite_vertical_abbreviation(self, subscript_offset=5, characters_max_distance=1.8):
        indices_to_remove = []
        one_vertical_abbreviation_postprocessed = False
        # Overwrite OCR boxes "S", "O", "2" vertically written
        for i_anchor, abbreviation_anchor in enumerate(self.abbreviations):
            
            anchor_height = abbreviation_anchor["box"][1][1] - abbreviation_anchor["box"][0][1]
            anchor_center = [
                (abbreviation_anchor["box"][0][0] + abbreviation_anchor["box"][1][0])//2,
                (abbreviation_anchor["box"][0][1] + abbreviation_anchor["box"][1][1])//2
            ]

            sulfur_dioxide_components = {"S": {"found": False}, "O": {"found": False}, "2": {"found": False}}
            difluoromethylene_components = {"C": {"found": False}, "F": {"found": False}, "2": {"found": False}}
            
            if (abbreviation_anchor["text"] == "O") or (abbreviation_anchor["text"] == "0"):
                sulfur_dioxide_components["O"] = { 
                    "found": True,
                    "box": abbreviation_anchor["box"],
                    "index": i_anchor
                }
            if (abbreviation_anchor["text"] == "O2") or (abbreviation_anchor["text"] == "02") or (abbreviation_anchor["text"] == "Oz") or (abbreviation_anchor["text"] == "0z"):
                sulfur_dioxide_components["O"] = { 
                    "found": True,
                    "box": abbreviation_anchor["box"],
                    "index": i_anchor
                }
                sulfur_dioxide_components["2"] = {
                    "found": True,
                    "box": abbreviation_anchor["box"], 
                    "index": i_anchor
                }
            
            if (abbreviation_anchor["text"] == "F"):
                difluoromethylene_components["F"] = { 
                    "found": True,
                    "box": abbreviation_anchor["box"],
                    "index": i_anchor
                }
            if (abbreviation_anchor["text"] == "F2") or (abbreviation_anchor["text"] == "Fz"):
                difluoromethylene_components["F"] = { 
                    "found": True,
                    "box": abbreviation_anchor["box"],
                    "index": i_anchor
                }
                difluoromethylene_components["2"] = {
                    "found": True,
                    "box": abbreviation_anchor["box"], 
                    "index": i_anchor
                }            
                
            if not(sulfur_dioxide_components["O"]["found"] or difluoromethylene_components["F"]["found"]):
                continue

            sulfur_dioxide_components_init = copy.deepcopy(sulfur_dioxide_components)
            difluoromethylene_components_init = copy.deepcopy(difluoromethylene_components)
            anchor_used = False
            for i, abbreviation in enumerate(self.abbreviations):
                center = [
                    (abbreviation["box"][0][0] + abbreviation["box"][1][0])//2,
                    (abbreviation["box"][0][1] + abbreviation["box"][1][1])//2
                ]
                if sulfur_dioxide_components["O"]["found"]:
                    # Check "S" 
                    if (abbreviation["text"] == "S"):
                        if math.dist(center, anchor_center) > characters_max_distance*anchor_height:
                            continue
                        sulfur_dioxide_components["S"] = {
                            "found": True,
                            "box": abbreviation["box"],
                            "index": i
                        }
                    # Check "2"
                    if ((abbreviation["text"] == "2") or (abbreviation["text"] == "z")) and ((center[1] + subscript_offset) >= anchor_center[1]):
                        if math.dist(center, anchor_center) > characters_max_distance*anchor_height:
                            continue
                        sulfur_dioxide_components["2"] = {
                            "found": True,
                            "box": abbreviation["box"], 
                            "index": i
                        }

                if difluoromethylene_components["F"]["found"]:
                    # Check "C" 
                    if (abbreviation["text"] == "C") or (abbreviation["text"] == "c") or (abbreviation["text"] == "O") or (abbreviation["text"] == "0"):
                        if math.dist(center, anchor_center) > characters_max_distance*anchor_height:
                            continue
                        difluoromethylene_components["C"] = {
                            "found": True,
                            "box": abbreviation["box"],
                            "index": i
                        }
                    # Check "2"
                    if ((abbreviation["text"] == "2") or (abbreviation["text"] == "z")) and ((center[1] + subscript_offset) >= anchor_center[1]):
                        if math.dist(center, anchor_center) > characters_max_distance*anchor_height:
                            continue
                        difluoromethylene_components["2"] = {
                            "found": True,
                            "box": abbreviation["box"], 
                            "index": i
                        }

                components_list = [sulfur_dioxide_components, difluoromethylene_components]
                
                component_found = False
                for components in components_list:
                    if not(all(components[k]["found"] for k in components.keys())):
                        continue 
                    component_found = True
                    one_vertical_abbreviation_postprocessed = True

                if not(component_found):
                    continue

                for components in components_list:
                    if not(all(components[k]["found"] for k in components.keys())):
                        continue 

                    # Create components box
                    min_x, min_y, max_x, max_y = float("inf"), float("inf"), -float("inf"), -float("inf")
                    for component in components.values():
                        if component["box"][0][0] < min_x:
                            min_x = component["box"][0][0]
                        if component["box"][0][1] < min_y:
                            min_y = component["box"][0][1]
                        if component["box"][1][0] > max_x:
                            max_x = component["box"][1][0]
                        if  component["box"][1][1] > max_y:
                            max_y = component["box"][1][1]
                    components_box = [[min_x, min_y], [max_x, max_y]]

                    atom_index_to_replace = None
                    for atom_index, atom in enumerate(self.molecular_graph.atoms): 
                        if not((atom["position"][0] >= (components_box[0][0])) and \
                            (atom["position"][0] <= (components_box[1][0])) and \
                            (atom["position"][1] >= (components_box[0][1])) and \
                            (atom["position"][1] <= (components_box[1][1]))):
                            continue
                        
                        # Find which atom inside the box has 2 connections going outside the box
                        neighbors = self.molecular_graph.get_neighbors(atom)
                        nb_external_connections = 0
                        for neighbor in neighbors:
                            if not((neighbor["position"][0] >= (components_box[0][0])) and \
                                (neighbor["position"][0] <= (components_box[1][0])) and \
                                (neighbor["position"][1] >= (components_box[0][1])) and \
                                (neighbor["position"][1] <= (components_box[1][1]))):
                                nb_external_connections += 1
                        if nb_external_connections != 2:
                            continue 
                        
                        # Check that the atom has only 2 single bonds connections
                        connected_to_non_single_bond = False
                        for neighboring_bond in self.molecular_graph.get_neighboring_bonds_from_atom(atom_index):
                            if ("DASHED" in self.molecular_graph.types_classes_bonds) and ("SOLID" in self.molecular_graph.types_classes_bonds):
                                if not((neighboring_bond["class"] == self.molecular_graph.types_classes_bonds["SINGLE"]) or \
                                       (neighboring_bond["class"] == self.molecular_graph.types_classes_bonds["DASHED"]) or
                                       (neighboring_bond["class"] == self.molecular_graph.types_classes_bonds["SOLID"])):
                                    connected_to_non_single_bond = True
                            else:
                                if not(neighboring_bond["class"] == self.molecular_graph.types_classes_bonds["SINGLE"]):
                                    connected_to_non_single_bond = True
                        if connected_to_non_single_bond:
                            continue 
                        
                        if "S" in components:
                            if (self.molecular_graph.atoms_classes_symbols[self.molecular_graph.atoms[atom_index]["class"]] != "S") and \
                               (self.molecular_graph.atoms_classes_symbols[self.molecular_graph.atoms[atom_index]["class"]] != "R"):
                                continue
                            
                        if "F" in components:
                            if (self.molecular_graph.atoms_classes_symbols[self.molecular_graph.atoms[atom_index]["class"]] != "C") and \
                               (self.molecular_graph.atoms_classes_symbols[self.molecular_graph.atoms[atom_index]["class"]] != "R"):
                                continue
                            
                        atom_index_to_replace = atom_index
                    
                    if atom_index_to_replace is None:
                        continue
                    
                    # Convert atom to R
                    self.molecular_graph.atoms[atom_index_to_replace]["class"] = self.molecular_graph.symbols_classes_atoms["R"]
                    
                    # Add new ocr box
                    if "S" in components:
                        self.abbreviations.append({
                            'text': "SO2", 
                            'box': [np.array([components_box[0][0], components_box[0][1]]), np.array([components_box[1][0], components_box[1][1]])], 
                            'score': 0.85
                        })
                        sulfur_dioxide_components = sulfur_dioxide_components_init
                    elif "F" in components:
                        self.abbreviations.append({
                            'text': "CF2", 
                            'box': [np.array([components_box[0][0], components_box[0][1]]), np.array([components_box[1][0], components_box[1][1]])], 
                            'score': 0.85
                        })
                        difluoromethylene_components = difluoromethylene_components_init
                 
                    for component in components.values():
                        if not("index" in component):
                            continue
                        
                        # Temporary mask used ocr boxes
                        self.abbreviations[component["index"]] = {
                            'text': "<mask>", 
                            'box': [np.array([0, 0]), np.array([0,0])], 
                            'score': 0
                        }
                        if (component["index"] in indices_to_remove):
                            continue 
                        indices_to_remove.append(component["index"])
                    anchor_used = True
                    if anchor_used:
                        break
                if anchor_used:
                    break
        # Remove ocr boxes
        for i_abb in sorted(indices_to_remove, reverse=True):
            self.abbreviations.pop(i_abb)
        self.postprocessing_flags["postprocessed_vertical_abbreviations"] = True 

    def postprocess_aromatic_bonds(self):
        for bond in self.molecule.GetBonds():
            try:
                if (str(bond.GetBondType()) == "AROMATIC") and (not bond.IsInRing()):
                    bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                    self.molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()).SetIsAromatic(False)
                    self.molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).SetIsAromatic(False)
            except Exception as e:
                print(f"{self.filename_logging}:Error in aromatic post-processing: {e}")
                pass 
        self.postprocessing_flags["postprocessed_aromatic_bonds"] = True
    
    def postprocess_aromatic_rings(self):
        ringInfo = self.molecule.GetRingInfo()
        bonds_rings = ringInfo.BondRings()
        for bonds_ring in bonds_rings:
            types = []
            for bond in bonds_ring:
                types.append(str(self.molecule.GetBondWithIdx(bond).GetBondType()))
            most_common_type = Counter(types).most_common(1)[0][0]
            if (most_common_type == "AROMATIC") and any([type != "AROMATIC" for type in types]):
                for bond in bonds_ring:
                    self.molecule.GetBondWithIdx(bond).SetBondType(Chem.rdchem.BondType.AROMATIC)
        self.postprocessing_flags["postprocessed_aromatic_rings"] = True
    
    def postprocess_polycycles_tailored(self):
        # Tailored post-processing to remove atoms 4 and 6 from the "C14C~CC2C1(C3C2(C3)[*])[S,C,O,N]4" polycycles 
        self.postprocessing_flags["postprocessed_polycycles_tailored"] = True 
        invalid_polycycle = Chem.MolFromSmarts("C14C~CC2C1(C3C2(C3)[*])[S,C,O,N]4") 
        if not(self.molecule.HasSubstructMatch(invalid_polycycle)):
            return 
        matched_atoms_indices_list = self.molecule.GetSubstructMatches(invalid_polycycle)     
        if len(matched_atoms_indices_list) != 1:
            return
        
        matched_atoms_indices = matched_atoms_indices_list[0]
        remove_atoms_indices = [matched_atoms_indices[4], matched_atoms_indices[6]]
        atoms_to_remove = [self.molecule.GetAtomWithIdx(matched_atoms_indices[4]), self.molecule.GetAtomWithIdx(matched_atoms_indices[6])]
        
        # Add bonds
        for _, atom_to_remove in zip(remove_atoms_indices, atoms_to_remove):
            neighbors = atom_to_remove.GetNeighbors()
            neighbors_matched = []
            center_position = np.array(self.molecular_graph.atoms[self.molecular_graph.get_atom_with_rdkit_idx(atom_to_remove.GetIdx())[1]]["position"])
            for i_1, neighbor_1 in enumerate(neighbors):
                if neighbor_1 in neighbors_matched:
                    continue
                neighbor_1_position = np.array(self.molecular_graph.atoms[self.molecular_graph.get_atom_with_rdkit_idx(neighbor_1.GetIdx())[1]]["position"])
                min_alignment_distance = float("inf")
                best_match_neighbor = None
                for i_2, neighbor_2 in enumerate(neighbors):
                    if (i_1 == i_2) or (neighbor_2 in neighbors_matched):
                        continue
                    neighbor_2_position = np.array(self.molecular_graph.atoms[self.molecular_graph.get_atom_with_rdkit_idx(neighbor_2.GetIdx())[1]]["position"])
                    alignment_distance = abs(np.cross(neighbor_1_position - center_position, neighbor_2_position - neighbor_1_position)/norm(center_position - neighbor_1_position))
                    if alignment_distance < min_alignment_distance:
                        min_alignment_distance = alignment_distance 
                        best_match_neighbor = neighbor_2

                molecule_editable = Chem.EditableMol(self.molecule)
                try:
                    molecule_editable.AddBond(neighbor_1.GetIdx(), best_match_neighbor.GetIdx(), order=Chem.rdchem.BondType.SINGLE)
                except Exception as e:
                    pass
                self.molecule = molecule_editable.GetMol()
                neighbors_matched.extend([neighbor_1, best_match_neighbor])
    
        # Remove atoms (in decreasing order)
        molecule_editable = Chem.EditableMol(self.molecule)
        for atom_index in sorted(remove_atoms_indices, reverse=True):
            molecule_editable.RemoveAtom(atom_index)
            self.molecular_graph.remove_atom_with_rdkit_idx(atom_index)
        self.molecule = molecule_editable.GetMol()
        
    def postprocess_polycycles(self):
        self.postprocessing_flags["postprocessed_polycycles"] = True 
        self.invalid_polycycles = [
            Chem.MolFromSmarts("C123C(C~CC1[S,C,O,N]2)C~C3"),
            Chem.MolFromSmarts("C123C(C~CC1C~C2)C~C3"),
            Chem.MolFromSmarts("C13C2(C(C~[S,C,O,N]~C1)C~[S,C,O,N]~C2)[S,C,O,N]3"),
            Chem.MolFromSmarts("C13[C]C4C[C]2(C1)C([C]C2C3)C4")
        ]
        for invalid_polycycle in self.invalid_polycycles:
            # Search invalid polycycle
            if not(self.molecule.HasSubstructMatch(invalid_polycycle)):
                continue
            matched_atoms_indices_list = self.molecule.GetSubstructMatches(invalid_polycycle)      

            remove_atoms_indices = []
            atoms_to_remove = []
            for matched_atoms_indices in matched_atoms_indices_list:
                # Search atom with 4 connection point
                for index in matched_atoms_indices:
                    atom = self.molecule.GetAtomWithIdx(index)
                    neighbors = atom.GetNeighbors()
                    if len(neighbors) != 4:
                        continue 
                    # Check that removing the atom does not create multiple fragments
                    molecule_editable = Chem.EditableMol(self.molecule)
                    molecule_editable.RemoveAtom(index)
                    test_molecule = molecule_editable.GetMol()
                    if len(Chem.GetMolFrags(test_molecule)) != 1:
                        continue
                    remove_atoms_indices.append(index)
                    atoms_to_remove.append(atom)

            if len(atoms_to_remove) > 1:
                print(f"{self.filename_logging}:The polycycle to post-proccess contains multiple 4-connection atom candidates. \
                    It is likely because of unexpected graph prediction.")
                break 
            
            # Add bonds
            for _, atom_to_remove in zip(remove_atoms_indices, atoms_to_remove):
                neighbors = atom_to_remove.GetNeighbors()
                neighbors_matched = []
                center_position = np.array(self.molecular_graph.atoms[self.molecular_graph.get_atom_with_rdkit_idx(atom_to_remove.GetIdx())[1]]["position"])
                for i_1, neighbor_1 in enumerate(neighbors):
                    if neighbor_1 in neighbors_matched:
                        continue
                    neighbor_1_position = np.array(self.molecular_graph.atoms[self.molecular_graph.get_atom_with_rdkit_idx(neighbor_1.GetIdx())[1]]["position"])
                    min_alignment_distance = float("inf")
                    best_match_neighbor = None
                    for i_2, neighbor_2 in enumerate(neighbors):
                        if (i_1 == i_2) or (neighbor_2 in neighbors_matched):
                            continue
                        neighbor_2_position = np.array(self.molecular_graph.atoms[self.molecular_graph.get_atom_with_rdkit_idx(neighbor_2.GetIdx())[1]]["position"])
                        alignment_distance = abs(np.cross(neighbor_1_position - center_position, neighbor_2_position - neighbor_1_position)/norm(center_position - neighbor_1_position))
                        if alignment_distance < min_alignment_distance:
                            min_alignment_distance = alignment_distance 
                            best_match_neighbor = neighbor_2

                    molecule_editable = Chem.EditableMol(self.molecule)
                    try:
                        molecule_editable.AddBond(neighbor_1.GetIdx(), best_match_neighbor.GetIdx(), order=Chem.rdchem.BondType.SINGLE)
                    except Exception as e:
                        pass
                    self.molecule = molecule_editable.GetMol()
                    neighbors_matched.extend([neighbor_1, best_match_neighbor])

            # Remove atoms (in decreasing order)
            molecule_editable = Chem.EditableMol(self.molecule)
            for atom_index in sorted(remove_atoms_indices, reverse=True):
                molecule_editable.RemoveAtom(atom_index)
                self.molecular_graph.remove_atom_with_rdkit_idx(atom_index)
            self.molecule = molecule_editable.GetMol()

    def resolve_abbreviations(self):
        self.matched_abbreviations_indices = []
        
        #draw_molecule_rdkit(smiles=Chem.MolToSmiles(self.molecule), molecule=self.molecule, augmentations=False, display_atom_indices=True, path="-1.png")
        for abbreviation_index, atom in enumerate(self.molecular_graph.atoms): 
            if not(self.molecular_graph.atoms_classes_symbols[atom["class"]] == "R"):
                continue
            for i, abbreviation in enumerate(self.abbreviations):
                if not((atom["position"][0] >= (abbreviation["box"][0][0])) and \
                    (atom["position"][0] <= (abbreviation["box"][1][0])) and \
                    (atom["position"][1] >= (abbreviation["box"][0][1])) and \
                    (atom["position"][1] <= (abbreviation["box"][1][1]))):
                    continue
                # Remove the matched abbreviation
                self.abbreviations = [abb for i_abb, abb in enumerate(self.abbreviations) if (i_abb != i)]

                if (abbreviation["text"] not in self.abbreviations_smiles_mapping) and (abbreviation["text"] not in self.ocr_atoms_classes_mapping):
                    #print("{self.filename_logging}Before correction: ", abbreviation["text"])
                    self.superatoms.append(abbreviation["text"])
                    abbreviation["text"] = self.spelling_corrector(abbreviation["text"])
                    #print("{self.filename_logging}:After correction: ", abbreviation["text"])
                    Chem.SetAtomAlias(self.molecule.GetAtomWithIdx(abbreviation_index), f"{abbreviation['text']}")
                    
                if abbreviation["text"] in self.ocr_atoms_classes_mapping:
                    #print(f"{self.filename_logging}:The OCR detection {abbreviation['text']} is replaced in the molecule.")
                    molecule_editable = Chem.EditableMol(self.molecule)
                    atom_symbol = self.ocr_atoms_classes_mapping[abbreviation["text"]]["symbol"]

                    # Atom with charge
                    if "," in atom_symbol:
                        atom_symbol, charge = atom_symbol.split(",")
                        rdkit_atom = Chem.Atom(atom_symbol)
                        rdkit_atom.SetFormalCharge(int(charge))
                        molecule_editable.ReplaceAtom(abbreviation_index, rdkit_atom)
                    elif ";" in atom_symbol:
                        atom_symbol, isotope_nb = atom_symbol.split(";")
                        rdkit_atom = Chem.Atom(atom_symbol)
                        rdkit_atom.SetIsotope(int(isotope_nb))
                        molecule_editable.ReplaceAtom(abbreviation_index, rdkit_atom)
                    else:
                        molecule_editable.ReplaceAtom(abbreviation_index, Chem.Atom(atom_symbol))
                    self.molecule = molecule_editable.GetMol()
                    continue
                
                #print(f"{self.filename_logging}:Abbreviation: {abbreviation['text']}, {self.abbreviations_smiles_mapping[abbreviation['text']]['smiles']}")  
                if abbreviation["text"] not in self.abbreviations_smiles_mapping:
                    self.molecule.GetAtomWithIdx(abbreviation_index).SetProp("atomLabel", f"[{abbreviation['text']}]") 
                    Chem.SetAtomAlias(self.molecule.GetAtomWithIdx(abbreviation_index), f"[{abbreviation['text']}]")
                    continue
                
                # Read abbreviation sub-molecule
                molecule_abbreviation = Chem.MolFromSmiles(self.abbreviations_smiles_mapping[abbreviation["text"]]["smiles"])
                if molecule_abbreviation is None:
                    print(f"{self.filename_logging}:Invalid abbreviation mapping: {abbreviation}, {self.abbreviations_smiles_mapping[abbreviation['text']]['smiles']}")
                    continue

                # Save molecule abbreviation connection points
                multiple_bonds_error = False
                molecule_abbreviation_connection_points = {}
                connection_point_abbreviation_index = 0
                for atom_index, connection_atom in enumerate(molecule_abbreviation.GetAtoms()):
                    if connection_atom.GetSymbol() == "*": 
                        bonds = connection_atom.GetBonds()
                        if len(bonds) > 1:
                            print(f"{self.filename_logging}:Error: connection atom from abbreviation has multiple bonds")
                            multiple_bonds_error = True
                            break
                        bond = bonds[0]
                        # Save connection atoms
                        for neighbor_index in [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]:
                            if molecule_abbreviation.GetAtomWithIdx(neighbor_index).HasProp(f"{abbreviation_index}-attachmentIndex"):
                                # Different atoms of the molecule can attach to the same position in the abbreviation
                                molecule_abbreviation.GetAtomWithIdx(neighbor_index).SetProp(
                                    f"{abbreviation_index}-attachmentIndex",
                                    molecule_abbreviation.GetAtomWithIdx(neighbor_index).GetProp(f"{abbreviation_index}-attachmentIndex") + "," + str(connection_point_abbreviation_index)
                                )
                            else:
                                molecule_abbreviation.GetAtomWithIdx(neighbor_index).SetProp(f"{abbreviation_index}-attachmentIndex", str(connection_point_abbreviation_index))
                            molecule_abbreviation_connection_points[str(connection_point_abbreviation_index)] = {
                                "bond": bond.GetBondType()
                            }
                        connection_point_abbreviation_index += 1

                if multiple_bonds_error:
                    # One connection atom attachs with multiple bond. (It should be impossible.)
                    continue

                nb_connection_points_abbreviation_molecule = connection_point_abbreviation_index
                
                # Remove extra atoms in molecule abbreviation
                molecule_abbreviation_editable = Chem.EditableMol(molecule_abbreviation)
                for atom_index in range(molecule_abbreviation.GetNumAtoms()-1, -1, -1):
                    if molecule_abbreviation.GetAtomWithIdx(atom_index).GetSymbol() == "*": 
                        molecule_abbreviation_editable.RemoveAtom(atom_index)
                molecule_abbreviation = molecule_abbreviation_editable.GetMol()
                
                # Retrieve connection points indices
                for atom_index, connection_atom in enumerate(molecule_abbreviation.GetAtoms()):
                    if connection_atom.HasProp(f"{abbreviation_index}-attachmentIndex"):
                        connection_point_abbreviation_indices = connection_atom.GetProp(f"{abbreviation_index}-attachmentIndex").split(",")
                        for connection_point_abbreviation_index in connection_point_abbreviation_indices:
                            molecule_abbreviation_connection_points[connection_point_abbreviation_index]["index"] = atom_index
        
                #draw_molecule_rdkit(smiles=Chem.MolToSmiles(self.molecule), molecule=self.molecule, augmentations=False, display_atom_indices=True, path = f"{abbreviation['text']}.png")
                # Save molecule connection points
                connection_point_index = 0
                for connection_atom in self.molecule.GetAtomWithIdx(abbreviation_index).GetNeighbors():
                    if connection_atom.HasProp("atomLabel") and (connection_atom.GetProp("atomLabel") == "abbreplaced"):
                        continue 
                    connection_point_index += 1
                nb_connection_points_molecule = connection_point_index
            
                #draw_molecule_rdkit(smiles=Chem.MolToSmiles(molecule_abbreviation), molecule=molecule_abbreviation, augmentations=False, display_atom_indices=True, path = f"pre-remove-abbreviation-{abbreviation['text']}.png")
                if nb_connection_points_molecule != nb_connection_points_abbreviation_molecule:
                    print(f"{self.filename_logging}:The number of connection point between the predicted abbreviation node and the associated sub-molecule \
                        ({abbreviation['text']} mapped to {self.abbreviations_smiles_mapping[abbreviation['text']]['smiles']}) mismatch")
                    break

                # Save molecule connection points
                connection_point_index = 0
                for connection_atom in self.molecule.GetAtomWithIdx(abbreviation_index).GetNeighbors():
                    if connection_atom.HasProp("atomLabel") and (connection_atom.GetProp("atomLabel") == "abbreplaced"):
                        continue 
                    
                    connection_atom.SetProp(f"{abbreviation_index}-attachmentIndex", str(connection_point_index))
                    connection_point_index += 1

                # Remove abbreviation in molecule
                self.matched_abbreviations_indices.append(abbreviation_index)
                
                # Retrieve connection points indices
                molecule_connection_points = {}
                for atom_index, connection_atom in enumerate(self.molecule.GetAtoms()):
                    if connection_atom.HasProp(f"{abbreviation_index}-attachmentIndex"):
                        connection_point_index = connection_atom.GetProp(f"{abbreviation_index}-attachmentIndex")
                        molecule_connection_points[connection_point_index] = {
                            "index": atom_index
                        }

                #draw_molecule_rdkit(smiles=Chem.MolToSmiles(molecule_abbreviation), molecule=molecule_abbreviation, augmentations=False, display_atom_indices=True, path = f"abbreviation-{abbreviation['text']}.png")
                offset = self.molecule.GetNumAtoms()
                # Combine
                self.molecule = Chem.CombineMols(self.molecule, molecule_abbreviation) 
                molecule_editable = Chem.EditableMol(self.molecule)

                # Add bonds (For multiple connection point abbreviations, the order should be from left to right, by checking bonds positions #TODO)
                for connection_point_index in molecule_connection_points.keys():
                    try:
                        molecule_editable.AddBond(
                            molecule_connection_points[connection_point_index]["index"], 
                            molecule_abbreviation_connection_points[connection_point_index]["index"] + offset, 
                            order=molecule_abbreviation_connection_points[connection_point_index]["bond"]
                        )
                    except Exception as e:
                        print(f'{self.filename_logging}:Error multiple connection points abbreviation ({connection_point_index}, {abbreviation}): {e}')
                        
                self.molecule = molecule_editable.GetMol()
                self.molecule.GetAtomWithIdx(abbreviation_index).SetProp("atomLabel", "abbreplaced")
                #draw_molecule_rdkit(smiles=Chem.MolToSmiles(self.molecule), molecule=self.molecule, augmentations=False, display_atom_indices=True, path = f"end-{abbreviation['text']}.png")
                break
            
        # Remove abbreviation in molecule (in decreasing order)
        molecule_editable = Chem.EditableMol(self.molecule)
        for abbreviation_index in sorted(self.matched_abbreviations_indices, reverse=True):
            molecule_editable.RemoveAtom(abbreviation_index)
        self.molecule = molecule_editable.GetMol()
        
        # Adjust wedge bonds indices after removing "abbreviation connections" atoms
        if self.assign_stereo:
            for bi in range(len(self.bonds_directions[0])):
                b, e = self.bonds_directions[0][bi]
                b -= sum([b > i for i in self.matched_abbreviations_indices])
                e -= sum([e > i for i in self.matched_abbreviations_indices])
                self.bonds_directions[0][bi] = b, e
        self.postprocessing_flags["postprocessed_abbreviations"] = True
    
    def remove_isolated_atoms(self):
        one_isolated_atom_postprocessed = False
        atoms_involved_in_connections = list(set([bond["index"][0] for bond in self.molecular_graph.bonds] + [bond["index"][1] for bond in self.molecular_graph.bonds]))
        removed_atom_indices = []
        for atom_idx in range(len(self.molecular_graph.atoms)):
            if atom_idx not in atoms_involved_in_connections:
                removed_atom_indices.append(atom_idx)
        removed_atom_indices = sorted(removed_atom_indices, reverse = True)

        if len(removed_atom_indices):
            for bond_idx in range(len(self.molecular_graph.bonds)):
                # Remove bonds connected to removed atoms
                b, e = self.molecular_graph.bonds[bond_idx]["index"]
                if b in removed_atom_indices or e in removed_atom_indices:
                    continue
                # Shift remaining atoms indices
                b -= sum([b > removed_atom_index for removed_atom_index in removed_atom_indices])
                e -= sum([e > removed_atom_index for removed_atom_index in removed_atom_indices])
                self.molecular_graph.bonds[bond_idx]["index"] = [b, e]
                
            atoms_new = []
            for atom_idx in range(len(self.molecular_graph.atoms)):
                if atom_idx not in removed_atom_indices:
                    atoms_new.append({
                    "index": self.molecular_graph.atoms[atom_idx]["index"],
                    "class": self.molecular_graph.atoms[atom_idx]["class"],
                    "position": self.molecular_graph.atoms[atom_idx]["position"],
                    "type": 1
                })
            self.molecular_graph.atoms = atoms_new
            one_isolated_atom_postprocessed = True

        self.postprocessing_flags["postprocessed_isolated_atoms"] = True
        if not(one_isolated_atom_postprocessed):
            return None
        return self.molecular_graph.to_rdkit(
            self.original_abbreviations, 
            self.abbreviations_smiles_mapping, 
            self.ocr_atoms_classes_mapping, 
            self.spelling_corrector, 
            align_rdkit_output=self.align_rdkit_output,
            remove_hydrogens=self.remove_hydrogens,
            postprocessing_flags=self.postprocessing_flags
        )
            
    def merge_keypoints(self):
        one_keypoints_postprocessed = False
        for i, abbreviation in enumerate(self.original_abbreviations):
            atoms_matched = []
            for abbreviation_index, atom in enumerate(self.molecular_graph.atoms): 
                if (atom["position"][0] >= (abbreviation["box"][0][0])) and \
                    (atom["position"][0] <= (abbreviation["box"][1][0])) and \
                    (atom["position"][1] >= (abbreviation["box"][0][1])) and \
                    (atom["position"][1] <= (abbreviation["box"][1][1])):
                    atoms_matched.append(atom)

            if len(atoms_matched) < 2:
                continue
            
            # If multiple atoms are located to the same ocr prediction location, trust the ocr. 
            # An alternative would be to look at the atoms predictions confidences
            if abbreviation["text"] in self.abbreviations_smiles_mapping:
                atom_class = self.molecular_graph.symbols_classes_atoms["R"]
            elif all(atom["class"] == atoms_matched[0]["class"] for atom in atoms_matched):
                atom_class = atoms_matched[0]["class"] #TODO Try to switch with R 
            else:
                # Get most represented prediction? most popular class?
                atom_class = self.molecular_graph.symbols_classes_atoms["R"] 
            
            # Create new atom
            new_atom_index = len(self.molecular_graph.atoms) 
            self.molecular_graph.atoms.append({
                "index": new_atom_index,
                "class": atom_class,
                "position": atoms_matched[0]["position"],
                "type": 1
            })

            # Update previous bonds
            remove_bonds_indices = []
            atoms_matched_indices = [atom_index for atom_index, atom in enumerate(self.molecular_graph.atoms) if atom in atoms_matched]
            for bond_index, bond in enumerate(self.molecular_graph.bonds):
                if (bond["index"][0] in atoms_matched_indices) and (bond["index"][1] in atoms_matched_indices):
                    remove_bonds_indices.append(bond_index)

                if bond["index"][0] in atoms_matched_indices:
                    new_index = [new_atom_index, bond["index"][1]]
                    if all((new_index != bond["index"]) for bond in self.molecular_graph.bonds) and all((list(reversed(new_index)) != bond["index"]) for bond in self.molecular_graph.bonds):
                        bond["index"] = new_index
                    else:
                        remove_bonds_indices.append(bond_index)
                    continue

                if bond["index"][1] in atoms_matched_indices:
                    new_index = [bond["index"][0], new_atom_index]
                    if all((new_index != bond["index"]) for bond in self.molecular_graph.bonds) and all((list(reversed(new_index)) != bond["index"]) for bond in self.molecular_graph.bonds):
                        bond["index"] = new_index
                    else:
                        remove_bonds_indices.append(bond_index)

            self.molecular_graph.bonds = [bond for bond_index, bond in enumerate(self.molecular_graph.bonds) if bond_index not in remove_bonds_indices]
        
            # Remove matched atoms
            for atom_matched in atoms_matched:
                self.molecular_graph.atoms.remove(atom_matched)
            
            # Update bond indices
            for i, bond in enumerate(self.molecular_graph.bonds):
                self.molecular_graph.bonds[i]["index"][0] -= sum(bond["index"][0] > removed_atom_index for removed_atom_index in atoms_matched_indices)
                self.molecular_graph.bonds[i]["index"][1] -= sum(bond["index"][1] > removed_atom_index for removed_atom_index in atoms_matched_indices)

            one_keypoints_postprocessed = True
        
        self.postprocessing_flags["postprocessed_keypoints"] = True   
        if not(one_keypoints_postprocessed):
            return None
        return self.molecular_graph.to_rdkit(
            self.original_abbreviations, 
            self.abbreviations_smiles_mapping, 
            self.ocr_atoms_classes_mapping, 
            self.spelling_corrector, 
            align_rdkit_output=self.align_rdkit_output,
            remove_hydrogens=self.remove_hydrogens,
            postprocessing_flags=self.postprocessing_flags
        )
             
    def assign_stereo_centers(self, molecule):
        molfile_path = "tmp.mol"
        edited_molfile_path = "tmp2.mol"
        self.bonds_directions[0] = [(bond[0] + 1, bond[1] + 1) for bond in self.bonds_directions[0]]
        rdmolfiles.MolToMolFile(molecule, molfile_path, kekulize=False)
        with open(edited_molfile_path, "w") as edited_molfile:
            with open(molfile_path, "r") as molfile:
                for l in molfile.readlines():
                    edit_chirality = False
                    sl = [c for c in l.strip().split(" ") if c != ""]
                    if len(sl) <= 2:
                        edited_molfile.write("  ".join(sl) + "\n")    
                        continue
                    try:
                        if sl[3] == "3":
                            sl[3] = "0"
                            edit_chirality = True
                        if (int(sl[0]),int(sl[1])) in self.bonds_directions[0]:
                            bond_index = self.bonds_directions[0].index((int(sl[0]),int(sl[1])))
                            if self.bonds_directions[1][bond_index] == "SOLID":
                                sl[3] = "1"
                            if self.bonds_directions[1][bond_index] == "DASHED":
                                sl[3] = "6"
                            edit_chirality = True
                    except Exception as e:
                        pass
                    if edit_chirality:
                        new_sl = []
                        for c in sl:
                            if len(c) == 2:
                                new_sl.append(" " + c)
                            if len(c) == 1:
                                new_sl.append("  " + c)
                        edited_molfile.write("".join(new_sl) + "\n")    
                    else:
                        edited_molfile.write(l)
        edited_molecule = Chem.MolFromMolFile(edited_molfile_path, sanitize=False, removeHs=False) 
        os.remove(molfile_path)
        os.remove(edited_molfile_path)
        if edited_molecule is None:
            print(f"{self.filename_logging}:Molecule with stereo-chemistry is None")
            return
        return edited_molecule
            
