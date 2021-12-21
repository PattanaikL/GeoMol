# %%
import os
from pathlib import Path, PosixPath
import argparse
from networkx.algorithms.coloring.greedy_coloring import STRATEGIES
import numpy as np
import pandas as pd

import torch
from rdkit.Chem import AllChem
from rdkit import  Geometry

import os
import pickle
import random
import copy

import time
# from model.model import GeoMol
from model.featurization import construct_loader
from generate_confs import generate_GeoMol_confs
# %%
def is_allzero(tensor):
    return False if torch.any(tensor != 0) else True

def set_rdmol_positions(mol, pose):
    for i in range(len(pose)):
        # mol.GetConformer(0).SetAtomPosition(i, pose[i].tolist())
        mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(float(pose[i,0]), float(pose[i,1]), float(pose[i,2])))
    return mol

def generate_conformers(mol, num_confs, smi):
    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    assert mol.GetNumConformers() == 0

    AllChem.EmbedMultipleConfs(
        mol, 
        numConfs=num_confs, 
        maxAttempts=0,
        ignoreSmoothingFailures=True,
    )
    if mol.GetNumConformers() != num_confs:
        print('Warning: Failure cases occured at %s, generated: %d , expected: %d.' % (smi, mol.GetNumConformers(), num_confs, ))
        return [mol, smi]
    
    return [mol]

def extract_smi_mols_ref(test_loader, args):
    smiles, K, mols, posss = [], [], [], []
    smi_mols_dict = {}
    testset = test_loader.dataset[:args.n_testset] if args.n_testset else test_loader.dataset
    for moldata in testset:
        mol = []
        molss = []
        smiles.append(moldata['name'])
        refnum = [is_allzero(moldata['pos'][0][i]) for i in range(args.n_true_confs)].count(False)
        K.append(refnum)
        mols.append(moldata['mol'])
        work_poss = [moldata['pos'][0][i] for i in range(args.n_true_confs) if not is_allzero(moldata['pos'][0][i])]
        posss.append(work_poss)
    
        for i, pose in enumerate(work_poss):
            mol = copy.deepcopy(moldata['mol']) # otherwise each mol will be changed to the last conf
            molss.append(set_rdmol_positions(mol, pose))
        
        smi_mols_dict[moldata['name']] = molss
    
    smi_conf_dict = {
        'smiles': smiles,
        'n_conformers': [ i for i in K ] # 2K will be announced by generate_confs.py
    }

    return smi_mols_dict, smi_conf_dict

def generate_rdkit_mols(smi_mols_dict):
    err_smis = []
    data_ref = smi_mols_dict
    smiles, confs = list(data_ref.keys()), list(data_ref.values())

    smi_rdkitmols_dict = {}
    for i in range(len(data_ref)):
        molss = []
        return_data = copy.deepcopy(confs[i])
        num_confs = len(return_data) * 2

        start_mol = return_data[0]
        mol_s = generate_conformers(start_mol, num_confs=num_confs, smi=smiles[i]) # The mol contains num_confs of confs
        if len(mol_s) != 1: # Error in generation
            mol, err_smi = mol_s
            err_smis.append(err_smi)
        else:
            mol = mol_s[0]

        num_pos_gen = mol.GetNumConformers()
        all_pos = []

        if num_pos_gen == 0:
            continue
        # assert num_confs == num_pos_gen
        for j in range(num_pos_gen):
            pose = mol.GetConformer(j).GetPositions()
            mol_new = copy.deepcopy(confs[i][0])
            molss.append(set_rdmol_positions(mol_new, pose))

        smi_rdkitmols_dict[smiles[i]] = molss

    return smi_rdkitmols_dict, err_smis

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type=PosixPath, help="name of dataset")
    # parser.add_argument("--split_path", type=PosixPath, help="name of dataset")
    parser.add_argument("--split", type=str, default="split0", help="split No.")
    parser.add_argument("--dataset", type=str, default="qm9", help="[drugs,qm9]")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_true_confs", type=int, default=10)
    parser.add_argument("--n_testset", type=int, help="number of mols in test set for evaluation, use all if not announced")
    parser.add_argument("--rdkit", action="store_true", default = False, help="whether to generate rdkit mols at the same time")
    parser.add_argument("--geomol", action="store_true", default = False, help="whether to generate GeoMol mols at the same time")
    parser.add_argument('--trained_model_dir', type=str, help="absolute path of trained model used for prediction")
    parser.add_argument('--mmff', action='store_true', default=False)
    parser.add_argument('--datatype', type=str, default='test', help="['train', 'val','test']")
    args = parser.parse_args()

    datetime = '-'.join(list(map(str, list(time.localtime())[1:5])))
    rootpath = Path(os.environ['HOME']) / "git-repo/AI-CONF/"
    split_path = rootpath / f"GeoMol/data/{args.dataset.upper()}/splits/{args.split}.npy"
    data_dir = rootpath / f"datasets/GeoMol/data/{args.dataset.upper()}/{args.dataset}"
    test_dir = rootpath / f"datasets/GeoMol/test/{args.dataset}-{args.split}/{datetime}"
    if not test_dir.exists():
        os.system(f"mkdir -p {test_dir}")

    seed = 2021
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    setattr(args, 'data_dir', data_dir)
    setattr(args, 'split_path', split_path)
    test_loader = construct_loader(args, modes=(args.datatype))
    
    smi_mols_dict, smi_conf_dict = extract_smi_mols_ref(test_loader, args)

    ### dump ref mols, rdkit mols, and test_smiles.csv ###

    with open(test_dir / "test_ref.pickle", 'wb') as f:
        pickle.dump(smi_mols_dict, f)
    df = pd.DataFrame(smi_conf_dict)
    df.to_csv(test_dir / "test_smiles.csv", index=False)
    ######################################################

    # """
    ################## RDKit generation ##################
    if args.rdkit:
        smi_rdkitmols_dict, err_smis = generate_rdkit_mols(smi_mols_dict)
        with open(test_dir / "test_rdkit.pickle", "wb") as fout:
            pickle.dump(smi_rdkitmols_dict, fout)
        with open(test_dir / "rdkit_err_smiles.txt", 'w') as f:
            f.write('\n'.join(err_smis))
            
    ################# GeoMol generation ##################
    # """
    # """
    if args.geomol:
        testdata = smi_conf_dict
        conformer_dict, err_smis = generate_GeoMol_confs(args, testdata)
        with open(test_dir / "test_GeoMol.pickle", "wb") as fout:
            pickle.dump(conformer_dict, fout)
        with open(test_dir / "GeoMol_err_smiles.txt", 'w') as f:
            f.write('\n'.join(err_smis))
    # """


    
