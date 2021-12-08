# %%
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import torch
from rdkit import Chem, Geometry

import pickle
import random
import copy

# from model.model import GeoMol
from model.featurization import construct_loader


class args(object):
    def __init__(self, data_dir, split_path, n_true_confs, dataset):
        self.data_dir = data_dir
        self.split_path = split_path
        self.n_true_confs = n_true_confs
        self.dataset = dataset
        self.batch_size = 50
        self.num_workers = 0

def is_allzero(tensor):
    return False if torch.any(tensor != 0) else True

def set_rdmol_positions(mol, pose):
    for i in range(len(pose)):
        # mol.GetConformer(0).SetAtomPosition(i, pose[i].tolist())
        mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(float(pose[i,0]), float(pose[i,1]), float(pose[i,2])))
    return mol
    
# %%
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", default="qm9", help="name of dataset")
    # parser.add_argument("--dataset", default="qm9", help="name of dataset")
    # args = parser.parse_args()

    split_path = Path("/pubhome/qcxia02/git-repo/AI-CONF/GeoMol/data/QM9/splits/split0.npy")
    data_dir = Path("/pubhome/qcxia02/git-repo/AI-CONF/GeoMol/data/QM9/qm9")
    n_true_confs = 10
    # dataset = args.dataset 
    dataset = "qm9"

    seed = 2021
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # test_split = np.load(split_path, allow_pickle=True)[2]
    config = args(data_dir, split_path, n_true_confs, dataset)
    test_loader = construct_loader(config, modes=('test'))
    
    # %%
    smiles, K, mols, posss = [], [], [], []
    smi_mols_dict = {}
    for moldata in test_loader.dataset[:100]:
        mol = []
        molss = []
        smiles.append(moldata['name'])
        refnum = [is_allzero(moldata['pos'][0][i]) for i in range(config.n_true_confs)].count(False)
        K.append(refnum)
        mols.append(moldata['mol'])
        work_poss = [moldata['pos'][0][i] for i in range(config.n_true_confs) if not is_allzero(moldata['pos'][0][i])]
        posss.append(work_poss)
    
        for i, pose in enumerate(work_poss):
            mol = copy.deepcopy(moldata['mol']) # otherwise each mol will be changed to the last conf
            molss.append(set_rdmol_positions(mol, pose))
        
        smi_mols_dict[moldata['name']] = molss
    
        # %%
    smi_conf_dict = {
        'smiles': smiles,
        'n_conformers': [ i for i in K ] # 2K will be announced by generate_confs.py
    }
    df = pd.DataFrame(smi_conf_dict)
    df.to_csv("tmp/test_smiles.csv", index=False)
    
    with open('tmp/test_ref.pickle', 'wb') as f:
        pickle.dump(smi_mols_dict, f)

# %%
    # 1. Then generate "test_gen.pickle" by generate_confs.py
    # 2. Then use get_results.py to do analysis