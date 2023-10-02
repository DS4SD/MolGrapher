# MolGrapher

This is the repository for [MolGrapher: Graph-based Visual Recognition of Chemical Structures](https://arxiv.org/abs/2308.12234).
The code will be released soon.

![MolGrapher](assets/model_architecture.png)

### Citation
```
@InProceedings{Morin_2023_ICCV,
    author    = {Morin, Lucas and Danelljan, Martin and Agea, Maria Isabel and Nassar, Ahmed and Weber, Valery and Meijer, Ingmar and Staar, Peter and Yu, Fisher},
    title     = {MolGrapher: Graph-based Visual Recognition of Chemical Structures},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {19552-19561}
}
```
### USPTO-30K Benchmark

USPTO-30K is available on [Hugging Face](https://huggingface.co/datasets/ds4sd/USPTO-30K).
- USPTO-10K contains 10,000 clean molecules, i.e. without any abbreviated groups. 
- USPTO-10K-abb contains 10,000 molecules with superatom groups.
- USPTO-10K-L contains 10,000 clean molecules with more than 70 atoms. 

### Synthetic Data

The synthetic dataset is available on [Hugging Face](https://huggingface.co/datasets/ds4sd/MolGrapher-Synthetic-300K).

