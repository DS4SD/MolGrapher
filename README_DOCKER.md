### Install MolGrapher via Docker

From the root of the repository, clone the [MolDepictor](https://github.com/DS4SD/MolDepictor/) repository.
```
cd MolGrapher
git clone git@github.com:DS4SD/MolDepictor.git
```

Build a Docker image
```
bash docker_build.sh
```

### Run MolGrapher via Docker 

Copy the images to process to `data/benchmarks/default/images`.

Run the image interactively
```
docker run -it --shm-size=2g molgrapher bash
```

Run MolGrapher
```
bash run.sh
```

Read predictions
```
less ../../../data/predictions/default/smiles.jsonl
```
