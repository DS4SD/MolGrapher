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
Links: [ICCV](https://openaccess.thecvf.com/content/ICCV2023/html/Morin_MolGrapher_Graph-based_Visual_Recognition_of_Chemical_Structures_ICCV_2023_paper.html), [Arxiv](https://arxiv.org/abs/2308.12234) 

### Installation

Create a virtual environment.
```
python3.11 -m venv molgrapher-env
source molgrapher-env/bin/activate
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

Models are available on [Hugging Face](https://huggingface.co/ds4sd/MolGrapher). They are automatically downloaded in `MolGrapher/data/models` when running the model. The model parameters are documented [here](https://github.com/DS4SD/MolGrapher/blob/b855f21567afced54c6ab680654c88ae5a40ef14/molgrapher/models/molgrapher_model.py#L42).

### Inference

#### Script
Your input images can be placed in the folder: `MolGrapher/data/benchmarks/default/`.
```
bash molgrapher/scripts/annotate/run.sh
```
Output predictions are saved in: `MolGrapher/data/predictions/default/`.

#### Python
```
from molgrapher.models.molgrapher_model import MolgrapherModel

model = MolgrapherModel()
images_or_paths = ["./data/benchmarks/default/images/image_0.png"] 
annotations = model.predict_batch(images_or_paths) 
```

`annotations` is a list of dictionnaries with fields:
```
[
    {
        'smi': 'O=C(O)C1=CC=C(C2=C(...',                      # MolGrapher SMILES prediction
        'conf': 0.991,                                        # MolGrapher confidence
        'file-info': {
            'filename': '...',                                # Input image filename
            'image_nbr': 1       
        }, 
        'abbreviations_ocr': [...],                           # Detected OCR text
        'abbreviations': [...],                               # Post-processed detected OCR text
        'annotator': {'program': 'MolGrapher', 'version': '1.0.0'},
   },
   ...
]
```

### Docling Integration
[Docling](https://github.com/DS4SD/docling) is a toolkit to extract the content and structure from PDF documents. It recognizes page layout, reading order, table structure, code, formulas, and classify images. 
Here, we combine `docling` and `MolGrapher`: 
- `Docling` segments and classify chemical-structure images from document pages,
- `MolGrapher` converts images to SMILES.

Install `docling` in the `molgrapher` environment.
```
pip install docling
```

Option 1. Convert a PDF document with `docling` and enrich it with `MolGrapher` annotations. 

Example: 
```
bash molgrapher/scripts/annotate/docling/docling_convert_and_enrich.sh ./data/pdfs/US9259003_page_4.pdf ./data/docling_documents/US9259003_page_4/
# bash [script] [pdf-path] [docling-document-directory-path]
```
Option 2. Enrich an existing `docling` document with `MolGrapher` annotations.

Example: 
```
python3 molgrapher/scripts/annotate/docling/enrich_docling_document.py --docling-document-directory-path ./data/docling_documents/US9259003_page_4/  
# python3 [script] --docling-document-directory-path [docling-document-directory-path]
```

The `docling` document, enriched with SMILES predictions, will be stored in `[docling-document-directory-path]`.
For more information, please refer to [docling](https://github.com/DS4SD/docling).

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
