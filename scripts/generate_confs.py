import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import random
import yaml
from argparse import ArgumentParser
from pathlib import Path, PosixPath
from multiprocessing import Pool

import torch
from torch_geometric.data import Batch
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import Get3DDistanceMatrix

from model.model import GeoMol
from model.featurization import featurize_mol_from_smiles
from model.inference import construct_conformers

class FooClass(object):
    def __init__(self, content):
        self.values = content

def pass_clash_filter(rdmol, cutoff):
    matrix3d = Get3DDistanceMatrix(rdmol)
    return False if ((matrix3d > 0) & (matrix3d < cutoff)).any() else True

def generate_GeoMol_confs(args, testdata):
    err_smis = []
    trained_model_dir = args.trained_model_dir
    dataset = args.dataset
    mmff = args.mmff
    cutoff = args.cutoff

    with open(f'{trained_model_dir}/model_parameters.yml') as f:
        model_parameters = yaml.full_load(f) # one can alter the noise std here
    model = GeoMol(**model_parameters)
    
    state_dict = torch.load(f'{trained_model_dir}/best_model.pt', map_location=torch.device('cpu'))
    # state_dict = torch.load(f'{trained_model_dir}/best_model.pt', map_location=torch.device('cuda'))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    conformer_dict = {}
    for smi, n_confs in tqdm(testdata.values):
        
        # create data object (skip smiles rdkit can't handle)
        tg_data = featurize_mol_from_smiles(smi, dataset=dataset) # Use RDKit to extract mol information first
        if not tg_data:
            print(f'failed to featurize SMILES: {smi}')
            err_smis.append(smi)
            continue
        
        # generate model predictions
        data = Batch.from_data_list([tg_data]) # batch comes from here?
        model(data, inference=True, n_model_confs=n_confs*2)
        
        # set coords
        n_atoms = tg_data.x.size(0)
        model_coords = construct_conformers(data, model)
        mols = []
        for x in model_coords.split(1, dim=1):
            mol = Chem.AddHs(Chem.MolFromSmiles(smi)) # hydrogens will be added to generated mols first by rdkit, to ensure the same total atoms
            coords = x.squeeze(1).double().cpu().detach().numpy()
            mol.AddConformer(Chem.Conformer(n_atoms), assignId=True)
            for i in range(n_atoms):
                mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
            if mmff:
                try:
                    AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
                except Exception as e:
                    pass
        # filter clashed confs
            if pass_clash_filter(mol, cutoff):
                mols.append(mol)
        ######################
            # mols.append(mol)
        if len(mols):
            conformer_dict[smi] = mols # only save smis that with cutoff > 0.7
        else:
            err_smis.append(smi)
    return conformer_dict, err_smis


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--trained_model_dir', type=str)
    parser.add_argument('--out', type=PosixPath)
    parser.add_argument('--test_csv', type=str)
    parser.add_argument('--dataset', type=str, default='qm9')
    parser.add_argument('--mmff', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--numgenconfs', type=int, default=5, help="default generating 10 confs each smi")
    parser.add_argument('--cutoff', type=float, default=0.7, help="simply filter molecules with clash")
    # parser.add_argument('--smi', type=str, help="smi for conformation generation")
    args = parser.parse_args()
    
    setattr(args, 'smi', "CC(C)=O")
    # setattr(args, 'smi', "C1CCCCC1")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.test_csv: # batch mode
        test_data = pd.read_csv(args.test_csv)
    elif args.smi and args.numgenconfs: # single mode
        test_data = FooClass([[args.smi, args.numgenconfs]])
    else:
        print("*"*40)
        print("bad input, please check your input, which should at least include args.smi & args.numgenconfs || args.test_csv")
        print("*"*40)
        quit()

    
    conformer_dict, err_smis = generate_GeoMol_confs(args, test_data)
    
    # save to file
    if args.out:
        with open(args.out, 'wb') as f:
            pickle.dump(conformer_dict, f)
    else:
        suffix = '_ff' if args.mmff else ''
        with open(f'{args.trained_model_dir}/test_mols{suffix}.pkl', 'wb') as f:
            pickle.dump(conformer_dict, f)
    
    with open(args.out.parent / "GeoMol_err_smiles.txt", 'w') as f:
        f.write('\n'.join(err_smis))
    