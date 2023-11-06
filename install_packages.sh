#!/bin/bash

# Install PyTorch
python3.9 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 \
    pytorch_lightning==2.0.2 torch-geometric==2.3.1 

# Install Other Libraries
python3.9 -m pip install scikit-learn seaborn timm mahotas more_itertools \
    rdkit-pypi CairoSVG SmilesPE python-Levenshtein nltk ipykernel ipython \
    rouge-score opencv-python albumentations torchsummary weighted-levenshtein \
    pytesseract shapely datasets
