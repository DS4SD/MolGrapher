# MolGrapher

[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-USPTO%0A30K-blue)](https://huggingface.co/datasets/ds4sd/USPTO-30K/)
[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MolGrapher%0ASynthetic%0A300K-blue)](https://huggingface.co/datasets/ds4sd/MolGrapher-Synthetic-300K)
[![arXiv](https://img.shields.io/badge/arXiv-2308.12234-919191.svg)](https://doi.org/10.48550/arXiv.2308.12234)
[![ICCV](https://img.shields.io/badge/Paper-iccv51070.2023.01791-b31b1b.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Morin_MolGrapher_Graph-based_Visual_Recognition_of_Chemical_Structures_ICCV_2023_paper.html)

This is the repository for [MolGrapher: Graph-based Visual Recognition of Chemical Structures](https://openaccess.thecvf.com/content/ICCV2023/html/Morin_MolGrapher_Graph-based_Visual_Recognition_of_Chemical_Structures_ICCV_2023_paper.html).

![MolGrapher](assets/model_architecture.png)

### Citation

If you find this repository useful, please consider citing:
```
@InProceedings{Morin_2023_ICCV,
    author = {Morin, Lucas and Danelljan, Martin and Agea, Maria Isabel and Nassar, Ahmed and Weber, Valery and Meijer, Ingmar and Staar, Peter and Yu, Fisher},
    title = {MolGrapher: Graph-based Visual Recognition of Chemical Structures},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2023},
    pages = {19552-19561}
}
```
Publication in [ICCV](https://openaccess.thecvf.com/content/ICCV2023/html/Morin_MolGrapher_Graph-based_Visual_Recognition_of_Chemical_Structures_ICCV_2023_paper.html) (DOI: https://doi.org/10.1109/iccv51070.2023.01791)

Publication in [Arxiv](https://arxiv.org/abs/2308.12234) (DOI: https://doi.org/10.48550/arXiv.2308.12234)

### Installation

Create a virtual environment.
```
conda create -n molgrapher python=3.11
conda activate molgrapher
```
Install [MolGrapher](https://github.com/DS4SD/MolGrapher/) and [MolDepictor](https://github.com/DS4SD/MolDepictor/) for CPU.
```
pip install -e .["cpu"]
```

Install [MolGrapher](https://github.com/DS4SD/MolGrapher/) and [MolDepictor](https://github.com/DS4SD/MolDepictor/) for GPU. (Tested for x86_64, Linux Ubuntu 20.04, CUDA 11.7, CUDNN 8.4)
```
pip install -e .["gpu"]
```
CUDA and CDNN versions can be edited in `setup.py`.

To install and run MolGrapher using Docker, please refer to [README_DOCKER.md](https://github.com/DS4SD/MolGrapher/blob/main/README_DOCKER.md).

### Model

Models are available on [Hugging Face](https://huggingface.co/ds4sd/MolGrapher).
```
wget https://huggingface.co/ds4sd/MolGrapher/resolve/main/models/graph_classifier/gc_gcn_model.ckpt -P ./data/models/graph_classifier/
wget https://huggingface.co/ds4sd/MolGrapher/resolve/main/models/graph_classifier/gc_no_stereo_model.ckpt -P ./data/models/graph_classifier/
wget https://huggingface.co/ds4sd/MolGrapher/resolve/main/models/graph_classifier/gc_stereo_model.ckpt -P ./data/models/graph_classifier/
wget https://huggingface.co/ds4sd/MolGrapher/resolve/main/models/keypoint_detector/kd_model.ckpt -P ./data/models/keypoint_detector/
```

After downloading, the folder `models` from Hugging Face should be placed in: `./data/`.
Models can be selected by modifying attributes of GraphRecognizer (in `./molgrapher/models/graph_recognizer.py`). 

### Inference

Your input images can be placed in the folder: `./data/benchmarks/default/`.
```
bash molgrapher/scripts/annotate/run.sh
```
Output predictions are saved in: `./data/predictions/default/`.

### USPTO-30K Benchmark

USPTO-30K is available on [Hugging Face](https://huggingface.co/datasets/ds4sd/USPTO-30K).
- USPTO-10K contains 10,000 clean molecules, i.e. without any abbreviated groups. 
- USPTO-10K-abb contains 10,000 molecules with superatom groups.
- USPTO-10K-L contains 10,000 clean molecules with more than 70 atoms. 

### Synthetic Dataset

The synthetic dataset is available on [Hugging Face](https://huggingface.co/datasets/ds4sd/MolGrapher-Synthetic-300K).
Images and graphs are generated using [MolDepictor](https://github.com/DS4SD/MolDepictor/).

### Training

To train the keypoint detector:
```
python3 ./molgrapher/scripts/train/train_keypoint_detector.py
```
To train the node classifier:
```
python3 ./molgrapher/scripts/train/train_graph_classifier.py
```
