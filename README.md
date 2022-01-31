# GeoMol: Torsional Geometric Generation of Molecular 3D Conformer Ensembles

---
This repository contains a method to generate 3D conformer ensembles directly from the molecular graph as described in
our [paper](https://arxiv.org/pdf/2106.07802.pdf). 


## Requirements

* python (version>=3.7.9)
* pytorch (version>=1.7.0)
* rdkit (version>=2020.03.2)
* pytorch-geometric (version>=1.6.3)
* networkx (version>=2.5.1)
* pot (version>=0.7.0)

## Installation

### Data
Download and extract the GEOM dataset from the original source:

1. `wget https://dataverse.harvard.edu/api/access/datafile/4327252`
2. `tar -xvf 4327252`

### Environment
Run `make conda_env` to create the conda environment. 
The script will request you to enter one of the supported CUDA versions listed [here](https://pytorch.org/get-started/locally/).
The script uses this CUDA version to install PyTorch and PyTorch Geometric. Alternatively, you could manually follow the
steps to install PyTorch Geometric [here](https://github.com/rusty1s/pytorch_geometric/blob/master/.travis.yml).

## Usage
This should result in two different directories, one for each half of GEOM. You should place the qm9 conformers directory
in the `data/QM9/` directory and do the same for the drugs directory. This is all you need to train the model:

`python train.py --data_dir data/QM9/qm9/ --split_path data/QM9/splits/split0.npy --log_dir ./test_run --n_epochs 250 --dataset qm9`

Use the provided script to generate conformers. The `test_csv` arg should be a csv file with SMILES in the first column,
and the number of conformers you want to generate in the second column. This will output a compressed dictionary of rdkit
mols in the `trained_model_dir` directory (unless you provide the `out` arg):

`python generate_confs.py --trained_model_dir trained_models/qm9/ --test_csv data/QM9/test_smiles.csv --dataset qm9`

However, note that to reproduce the numbers in our paper, one needs additionally to run scripts/clean_smiles.py to account for inconsistent molecules in the dataset. See also count_geomol_failures.ipynb .
You can use the provided `visualize_confs.ipynb` jupyter notebook to visualize the generated conformers.

## Additional comments

### Training
To train the model, our code randomly samples files from the GEOM dataset and randomly samples conformers within those
files. This is a lot of file I/O, which wasn't a huge issue for us when training, but could be an issue for others. If
you're having issues with this, feel free to reach out, and I can help you reconfigure the code.

### Some limitations
Currently, the model is hardcoded for atoms with a max of 4 neighbors. Since the dataset we train on didn't have atoms
with more than 4 neighbors, we made this choice to speed up the code. In principle, the code can be adapted for something
like a pentavalent phosphorus, but this wasn't a priority for us.

We can't deal with disconnected fragments (i.e. there is a "." in the SMILES).

This code will work poorly for macrocycles.

To ensure correct predictions, ALL tetrahedral chiral centers must be specified. There's probably a way to automate the
specification of "rigid" chiral centers (e.g. in a fused ring), which I'll hopefully figure out soon, but I'm grad
student with limited time :(

### Feedback and collaboration
Code like this doesn't improve without feedback from the community. If you have comments/suggestions, please reach out
to us! We're always happy to chat and provide input on how you can take this method to the next level.

