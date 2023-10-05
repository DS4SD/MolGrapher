#!/bin/bash

# Update conda
conda update -n base -c defaults conda
conda install pip 

# Install conda packages
conda install -c pytorch -y pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 
conda install -c pyg -y pyg 
conda install -c conda-forge -y pytorch-lightning
conda install -y pandas jupyterlab scikit-learn matplotlib tqdm
conda install -c anaconda -y seaborn
conda install -c conda-forge -y timm
conda install -c conda-forge -y mahotas

# Pip install
pip install rdkit-pypi CairoSVG SmilesPE python-Levenshtein nltk ipykernel ipython rouge-score opencv-python albumentations paddleocr paddlepaddle torchsummary weighted-levenshtein pytesseract shapely

# Optionally, set up OSRA, Imago, MolVec, DECIMER, Img2Mol, MolScribe, CEDe

