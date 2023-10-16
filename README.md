# MolGrapher

[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-USPTO%0A30K-blue)](https://huggingface.co/datasets/ds4sd/USPTO-30K/)
[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MolGrapher%0ASynthetic%0A300K-blue)](https://huggingface.co/datasets/ds4sd/MolGrapher-Synthetic-300K)

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

### Installation

Install [MolGrapher](https://github.com/DS4SD/MolGrapher/)
```
conda create -n molgrapher python=3.9
bash install_packages.sh
pip install -e .
```

Install [MolDepictor](https://github.com/DS4SD/MolDepictor/)
```
git clone https://github.com/DS4SD/MolDepictor.git
cd MolDepictor
pip install -e .
```

### Model

Models are available on [Hugging Face](https://huggingface.co/ds4sd/MolGrapher).

After downloading, the folder: `models`, should be placed in: `./data/models/graph_classifier` and `./data/models/keypoint_detector`.
Models can be selected by modifying attributes of GraphRecognizer (in `./molgrapher/models/graph_recognizer.py`). 

### Inference

Your input images can be placed in the folder: `./data/benchmarks/default/`.
```
bash molgrapher/scripts/annotate/run.sh
```
Output predictions are saved in: `./data/predictions/default/`.

By default, molgrapher runs on CPU. This option can be modified in: `./molgrapher/scripts/annotate/run_predict.sh`, changing the flag `--force-cpu` to `--no-force-cpu`.

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
