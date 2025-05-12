This repo is a dataloader tool based on LaminDB for training large-scale models using large amount of data distributed on many Anndata h5ad files.


Installation
------------

1. Create the conda/mamba environment:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate lamin-dataloader
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Setup a lamindb instance [according to the instructions](https://docs.lamin.ai/setup)