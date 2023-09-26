# MolGrapher

This is the repository for [MolGrapher: Graph-based Visual Recognition of Chemical Structures](https://arxiv.org/abs/2308.12234).
The code will be released soon.

![MolGrapher](assets/model_architecture.png)

### Citation
```
@inproceedings{morin2023molgrapher,
    title={MolGrapher: Graph-based Visual Recognition of Chemical Structures}, 
    author={Lucas Morin and Martin Danelljan and Maria Isabel Agea and Ahmed Nassar and Valery Weber and Ingmar Meijer and Peter Staar and Fisher Yu},
    journal={International Conference on Computer Vision (ICCV)},
    year={2023}
}
```
### USPTO-30K Benchmark

USPTO-30K is available on [Hugging Face](https://huggingface.co/datasets/ds4sd/USPTO-30K).
- USPTO-10K contains 10,000 clean molecules, i.e. without any abbreviated groups. 
- USPTO-10K-abb contains 10,000 molecules with superatom groups.
- USPTO-10K-L contains 10,000 clean molecules with more than 70 atoms. 

### Synthetic Data

The synthetic dataset is available on [Hugging Face](https://huggingface.co/datasets/ds4sd/MolGrapher-Synthetic-300K).

