import argparse
import pickle

import copy
import numpy as np
from tqdm import tqdm

import rdkit
from rdkit.Chem import AllChem
from rdkit import Chem, Geometry

import multiprocessing
from functools import partial 

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
        # print('Warning: Failure cases occured at %s, generated: %d , expected: %d.' % (Chem.MolToSmiles(mol), mol.GetNumConformers(), num_confs, )) # Not compatible because of extra hydrogen in smiles from RDkit
        print('Warning: Failure cases occured at %s, generated: %d , expected: %d.' % (smi, mol.GetNumConformers(), num_confs, ))


    return mol

def set_rdmol_positions(mol, pose):
    for i in range(len(pose)):
        # mol.GetConformer(0).SetAtomPosition(i, pose[i].tolist())
        mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(float(pose[i,0]), float(pose[i,1]), float(pose[i,2])))
    return mol

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Get RDkit results")
    parser.add_argument('--ref', type=str, help="absolute path of referenced confs' pickle", required=True)
    parser.add_argument('--rdkit', type=str, help="absolute path of rdkit confs' pickle", required=True)
    args = parser.parse_args()

    with open(args.ref, 'rb') as f:
        data_ref = pickle.load(f)
    # data_ref: {smi:mol, smi:mol}
    smiles, confs = list(data_ref.keys()), list(data_ref.values())

    smi_rdkitmols_dict = {}
    for i in range(len(data_ref)):
        molss = []
        return_data = copy.deepcopy(confs[i])
        num_confs = len(return_data) * 2

        start_mol = return_data[0]
        mol = generate_conformers(start_mol, num_confs=num_confs, smi=smiles[i]) # The mol contains num_confs of confs
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

    with open(args.rdkit, "wb") as fout:
        pickle.dump(smi_rdkitmols_dict, fout)
    print('save generated conf to %s done!' % args.rdkit)
    