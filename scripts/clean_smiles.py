from rdkit import Chem

import os.path as osp
from statistics import mode, StatisticsError
import numpy as np
import pandas as pd
import pickle

# location of unzipped GEOM dataset
true_confs_dir = '../data/DRUGS/drugs'


def clean_confs(smi, confs):
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)
    return [confs[i] for i in good_ids]


def correct_smiles(true_confs):

    conf_smis = []
    for c in true_confs:
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c))
        conf_smis.append(conf_smi)

    try:
        common_smi = mode(conf_smis)
    except StatisticsError:
        return None  # these should be cleaned by hand

    if np.sum([common_smi == smi for smi in conf_smis]) == len(conf_smis):
        return mode(conf_smis)
    else:
        print('consensus', common_smi)  # these should probably also be investigated manually
        return common_smi
        # return None


test_data = pd.read_csv('../data/DRUGS/test_smiles_raw.csv')
corrected_smiles_dict = {}
failed_smiles = []
for i, data in test_data.iterrows():

    smi = data.smiles

    try:
        with open(osp.join(true_confs_dir, smi.replace('/', '_') + '.pickle'), "rb") as f:
            mol_dic = pickle.load(f)
    except FileNotFoundError:
        print(f'cannot find ground truth conformer file: {smi}')
        continue

    true_confs = [conf['rd_mol'] for conf in mol_dic['conformers']]
    true_confs = clean_confs(smi, true_confs)
    if len(true_confs) == 0:
        corrected_smiles_dict[smi] = smi
        continue

    corrected_smi = correct_smiles(true_confs)
    corrected_smiles_dict[smi] = corrected_smi

    if corrected_smi is None:
        failed_smiles.append(smi)
        corrected_smiles_dict[smi] = smi
        print(f'failed: {smi}\n')

test_data['corrected_smiles'] = list(corrected_smiles_dict.values())
test_data.to_csv('../data/DRUGS/test_smiles_corrected.csv', index=False)
