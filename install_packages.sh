#!/bin/bash

# Pip install
conda install -c pytorch -y pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 
conda install -c pyg -y pyg 
pip install torch_geometric pytorch-lightning pandas jupyterlab scikit-learn matplotlib tqdm seaborn timm mahotas more_itertools tensorboard
pip install rdkit-pypi CairoSVG SmilesPE python-Levenshtein nltk ipykernel ipython rouge-score opencv-python albumentations paddleocr paddlepaddle torchsummary weighted-levenshtein pytesseract shapely datasets

# For benchmarking, set up OSRA, Imago, MolVec, DECIMER, Img2Mol, MolScribe, CEDe

