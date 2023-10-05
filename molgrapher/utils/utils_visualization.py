#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem 
from tqdm import tqdm 


def plot_confusion_matrix(confusion_matrix, vocabulary):
    mask = sum(confusion_matrix) != 0
    confusion_matrix = confusion_matrix[mask][:, mask]
    classes = list(np.array(list(vocabulary.keys()))[mask])
    
    accuracies = confusion_matrix/confusion_matrix.sum(1)
    accuracies[accuracies > 1] = 1
    accuracies[accuracies < 0.005] = 0

    fig, ax = plt.subplots(figsize=(10, 8))
    cb = ax.imshow(accuracies, cmap='Reds')
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            color='white' if accuracies[j,i] > 0.5 else 'black'
            ax.annotate(f'{confusion_matrix[j,i]}', (i,j), 
                        color=color, va='center', ha='center')

    plt.colorbar(cb, ax=ax)
    plt.xlabel("Predictions")
    plt.ylabel("Ground truths")
    plt.show()
    plt.close()

def get_bin_index(value, bins):
    index = 1
    bin = bins[index]
    while value > bin:
        if value > bins[-1]:
            return len(bins) - 1
        print(index)
        index += 1
        bin = bins[index]
    return index 

def size_readjust(row, df, col_name):
    min_thr = int(df[col_name].mean() - df[col_name].std())
    max_thr = int(df[col_name].mean() + df[col_name].std())
    if row[col_name] <= min_thr:
        return 'Small (<' + str(min_thr) + ')'
    if row[col_name] < max_thr and  row[col_name] > min_thr:
        return 'Medium (' + str(min_thr)+'-' + str(max_thr)+')'
    if row[col_name] >= max_thr:
        return 'Big (>' + str(max_thr) + ')'

def piechart_function(df, inner_col, outer_col, title="Nested pie chart"):
    fig, ax = plt.subplots(figsize=(24, 12))
    size = 0.3

    outer = df.groupby(outer_col).sum()
    inner = df.groupby([outer_col, inner_col]).sum()
    inner_labels = inner.index.get_level_values(1)

    cmap = plt.colormaps["tab20c"]
    outer_colors = cmap(np.arange(3)*4)
    inner_colors = cmap([1, 2, 3, 5, 6, 7, 9, 10, 11])

    ax.pie(
        outer.values.flatten(), 
        radius=1,
        colors = outer_colors,
        labels = outer.index,
        autopct = '%1.1f%%',
        pctdistance = 0.85,
        labeldistance = 1.0,
        wedgeprops = dict(width=size, edgecolor='w')
    )

    ax.pie(
        inner.values.flatten(), 
        radius = (1-size),
        colors = inner_colors,
        labels = inner_labels,
        autopct = '%1.1f%%',
        pctdistance = 0.45,
        labeldistance = 0.6,
        wedgeprops = dict(width=size, edgecolor='w')
    )

    ax.set(aspect="equal", title='Pie plot with `ax.pie`')
    plt.title(title)
    plt.show()

def get_atoms_population(smiles_list=None, molfiles_paths=None):
    """
    This function takes a list of SMILES and computes the population of each atom class. 
    It returns two lists, the atoms and their population sharing index relation.
  
    Args:
        smiles_list (list): list of SMILES (str).

    Returns:
        sorted_population (list): list of integers containing the atom representation from the most to the least
                                  represented one.
        sorted_atoms (list): list of strings containing the atom representation from the most to the least
                             represented one.
    """
    # Get the atoms and their representation
    atoms = []
    population = []

    if molfiles_paths:
        for molfile_path in molfiles_paths:
            molecule = Chem.MolFromMolFile(molfile_path)
            if not molecule:
                continue
            for atom in molecule.GetAtoms():
                symbol = atom.GetSymbol()

                if atom.GetFormalCharge():
                    if atom.GetFormalCharge() > 0:
                        symbol = atom.GetSymbol() + ",+" + str(atom.GetFormalCharge())
                    else:
                        symbol = atom.GetSymbol() + "," + str(atom.GetFormalCharge())
                
                # Filter markush structures #TODO
                r_groups = ["X", "Y", "U", "W", "V"]
                if symbol in r_groups:
                    continue 

                if symbol not in atoms:
                    atoms.append(symbol)
                    population.append(1)
                else:
                    indx = atoms.index(symbol)
                    population[indx] = population[indx] + 1

    else:
        for smi in smiles_list:
            molecule = Chem.MolFromSmiles(smi)
            for atom in molecule.GetAtoms():
                symbol = atom.GetSymbol()

                if atom.GetFormalCharge():
                    symbol = atom.GetSymbol() + "," + str(atom.GetFormalCharge())
                
                if symbol not in atoms:
                    atoms.append(symbol)
                    population.append(1)
                else:
                    indx = atoms.index(symbol)
                    population[indx] = population[indx] + 1
    
    # Sort the lists from the highest to the lowest number.
    sorted_population, sorted_atoms = zip(*sorted(zip(population, atoms), reverse = True))
    
    return sorted_atoms, sorted_population