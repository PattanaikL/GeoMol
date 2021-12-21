import argparse
from pathlib import PosixPath
import pickle

import copy
import numpy as np
from tqdm import tqdm

import rdkit
from rdkit.Chem import AllChem
from rdkit import Chem, Geometry

import multiprocessing
from functools import partial 

def generate_conformers(mol, num_confs, smi, rmsd_thres, seed):
    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    assert mol.GetNumConformers() == 0

    AllChem.EmbedMultipleConfs(
        mol, 
        numConfs=num_confs, 
        maxAttempts=0,
        ignoreSmoothingFailures=True,
        randomSeed = seed,
        pruneRmsThresh = rmsd_thres

    )
    if mol.GetNumConformers() != num_confs:
        print('Warning: Failure cases occured at %s, generated: %d , expected: %d.' % (smi, mol.GetNumConformers(), num_confs, ))
        return [mol, smi]
    
    return [mol]

def set_rdmol_positions(mol, pose):
    for i in range(len(pose)):
        # mol.GetConformer(0).SetAtomPosition(i, pose[i].tolist())
        mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(float(pose[i,0]), float(pose[i,1]), float(pose[i,2])))
    return mol

def generate_rdkit_mols(data_ref, numconfs, rmsd_thres, seed):
    err_smis = []
    smiles, confs = list(data_ref.keys()), list(data_ref.values())

    smi_rdkitmols_dict = {}
    for i in range(len(data_ref)):
        molss = []
        return_data = copy.deepcopy(confs[i])
        num_confs = numconfs * 2

        # start_mol = return_data[0]
        start_mol = return_data

        mol_s = generate_conformers(start_mol, num_confs=num_confs, smi=smiles[i], rmsd_thres=rmsd_thres, seed=seed) # The mol contains num_confs of confs
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
            mol_new = copy.deepcopy(confs[i])
            molss.append(set_rdmol_positions(mol_new, pose))

        smi_rdkitmols_dict[smiles[i]] = molss

    return smi_rdkitmols_dict, err_smis

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Get RDkit results")
    parser.add_argument('--ref', type=PosixPath, help="absolute path of referenced confs' pickle", required=True)
    parser.add_argument('--rdkit', type=PosixPath, help="absolute path of rdkit confs' pickle", required=True)
    parser.add_argument("--numconfs", type=int, help="num_confs for rdkit generation")
    parser.add_argument("--rmsdt", type=float, help="rmsd threshold for rdkit conf generation")
    args = parser.parse_args()

    seed = 2021

    with open(args.ref, 'rb') as f:
        data_ref = pickle.load(f)
    # data_ref: {smi:mol, smi:mol}
    # smi_rdkitmols_dict, err_smis = generate_rdkit_mols(data_ref)
    smi_rdkitmols_dict, err_smis = generate_rdkit_mols(data_ref, args.numconfs, args.rmsdt, seed)


    with open(args.rdkit, "wb") as fout:
        pickle.dump(smi_rdkitmols_dict, fout)
    print('save generated conf to %s done!' % args.rdkit)

    with open(args.rdkit.parent / f"rdkit_err_smiles_{args.numconfs}_{args.rmsdt}.txt", 'w') as f:
        f.write('\n'.join(err_smis))