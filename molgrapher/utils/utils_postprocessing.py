#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdkit import Chem 
from rdkit.Chem import rdmolfiles
import os 


def assign_stereo_centers(molecule, bonds_directions):
    molfile_path = "tmp.mol"
    edited_molfile_path = "tmp2.mol"
    bonds_directions[0] = [(bond[0] + 1, bond[1] + 1) for bond in bonds_directions[0]]
    
    rdmolfiles.MolToMolFile(molecule, molfile_path, kekulize=False)
    
    with open(edited_molfile_path, "w") as edited_molfile:
        with open(molfile_path, "r") as molfile:
            for l in molfile.readlines():
                edit_chirality = False
                sl = [c for c in l.strip().split(" ") if c != ""]
                if len(sl) > 2:
                    try:
                        if sl[3] == "3":
                            sl[3] = "0"
                            edit_chirality = True

                        if (int(sl[0]),int(sl[1])) in bonds_directions[0]:
                            bond_index = bonds_directions[0].index((int(sl[0]),int(sl[1])))
                            if bonds_directions[1][bond_index] == "SOLID":
                                sl[3] = "1"

                            if bonds_directions[1][bond_index] == "DASHED":
                                sl[3] = "6"

                            edit_chirality = True
                    except Exception as e:
                        #print(e)
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
    return edited_molecule
    