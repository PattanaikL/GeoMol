from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType

import os.path as osp
import numpy as np
import glob
import pickle
import random

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Dataset, Data, DataLoader
from model.utils import get_dihedral_pairs

dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}


def one_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


class geom_confs(Dataset):
    def __init__(self, root, split_path, mode, transform=None, pre_transform=None, max_confs=10):
        super(geom_confs, self).__init__(root, transform, pre_transform)

        self.root = root
        self.split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        self.split = np.load(split_path, allow_pickle=True)[self.split_idx]
        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        # try:
        #     with open(osp.join(self.root, 'all_data.pickle'), 'rb') as f:
        #         data_dict = pickle.load(f)
        #     smiles = [list(data_dict)[i] for i in self.split]
        #     self.pickle_files = [data_dict[smi] for smi in smiles]
        # except FileNotFoundError:
        self.dihedral_pairs = {} # for memoization
        all_files = sorted(glob.glob(osp.join(self.root, '*.pickle')))
        self.pickle_files = [f for i, f in enumerate(all_files) if i in self.split]
        self.max_confs = max_confs

    def len(self):
        # return len(self.pickle_files)  # should we change this to an integer for random sampling?
        return 10000 if self.split_idx == 0 else 1000

    def get(self, idx):
        data = None
        while not data:
            pickle_file = random.choice(self.pickle_files)
            mol_dic = self.open_pickle(pickle_file)
            data = self.featurize_mol(mol_dic)

        if idx in self.dihedral_pairs:
            data.edge_index_dihedral_pairs = self.dihedral_pairs[idx]
        else:
            data.edge_index_dihedral_pairs = get_dihedral_pairs(data.edge_index, data=data)

        return data

    def open_pickle(self, mol_path):
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        return dic

    def featurize_mol(self, mol_dic):
        confs = mol_dic['conformers']
        random.shuffle(confs)  # shuffle confs
        name = mol_dic["smiles"]

        # filter mols rdkit can't intrinsically handle
        mol_ = Chem.MolFromSmiles(name)
        if mol_:
            canonical_smi = Chem.MolToSmiles(mol_)
        else:
            return None

        # skip conformers with fragments
        if '.' in name:
            return None

        # skip conformers without dihedrals
        N = confs[0]['rd_mol'].GetNumAtoms()
        if N < 4:
            return None
        if confs[0]['rd_mol'].GetNumBonds() < 4:
            return None
        if not confs[0]['rd_mol'].HasSubstructMatch(dihedral_pattern):
            return None

        pos = torch.zeros([self.max_confs, N, 3])
        pos_mask = torch.zeros(self.max_confs, dtype=torch.int64)
        k = 0
        for conf in confs:
            mol = conf['rd_mol']

            # skip mols with atoms with more than 4 neighbors for now
            n_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
            if np.max(n_neighbors) > 4:
                continue

            # filter for conformers that may have reacted
            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception as e:
                continue

            if conf_canonical_smi != canonical_smi:
                continue

            pos[k] = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
            pos_mask[k] = 1
            k += 1
            correct_mol = mol
            if k == self.max_confs:
                break

        # return None if no non-reactive conformers were found
        if k == 0:
            return None

        type_idx = []
        atomic_number = []
        atom_features = []
        chiral_tag = []
        neighbor_dict = {}
        ring = correct_mol.GetRingInfo()
        for i, atom in enumerate(correct_mol.GetAtoms()):
            type_idx.append(self.types[atom.GetSymbol()])
            n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(n_ids) > 1:
                neighbor_dict[i] = torch.tensor(n_ids)
            chiral_tag.append(chirality[atom.GetChiralTag()])
            atomic_number.append(atom.GetAtomicNum())
            atom_features.extend([atom.GetAtomicNum(),
                                  1 if atom.GetIsAromatic() else 0])
            atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
            atom_features.extend(one_k_encoding(atom.GetHybridization(), [
                                    Chem.rdchem.HybridizationType.SP,
                                    Chem.rdchem.HybridizationType.SP2,
                                    Chem.rdchem.HybridizationType.SP3,
                                    Chem.rdchem.HybridizationType.SP3D,
                                    Chem.rdchem.HybridizationType.SP3D2]))
            atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
            atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
            atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                                  int(ring.IsAtomInRingOfSize(i, 4)),
                                  int(ring.IsAtomInRingOfSize(i, 5)),
                                  int(ring.IsAtomInRingOfSize(i, 6)),
                                  int(ring.IsAtomInRingOfSize(i, 7)),
                                  int(ring.IsAtomInRingOfSize(i, 8))])
            atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))

        z = torch.tensor(atomic_number, dtype=torch.long)
        chiral_tag = torch.tensor(chiral_tag, dtype=torch.float)

        row, col, edge_type, bond_features = [], [], [], []
        for bond in correct_mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [self.bonds[bond.GetBondType()]]
            bt = tuple(sorted([bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()])), bond.GetBondTypeAsDouble()
            bond_features += 2 * [int(bond.IsInRing()),
                                  int(bond.GetIsConjugated()),
                                  int(bond.GetIsAromatic())]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(self.bonds)).to(torch.float)
        # bond_features = torch.tensor(bond_features, dtype=torch.float).view(len(bond_type), -1)

        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
        # edge_attr = torch.cat([edge_attr[perm], bond_features], dim=-1)
        edge_attr = edge_attr[perm]

        row, col = edge_index
        hs = (z == 1).to(torch.float)
        num_hs = scatter(hs[row], col, dim_size=N).tolist()

        x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(self.types))
        x2 = torch.tensor(atom_features).view(N, -1)
        x = torch.cat([x1.to(torch.float), x2], dim=-1)

        data = Data(x=x, z=z, pos=[pos], edge_index=edge_index, edge_attr=edge_attr, neighbors=neighbor_dict,
                    chiral_tag=chiral_tag, name=name, boltzmann_weight=conf['boltzmannweight'],
                    degeneracy=conf['degeneracy'], mol=correct_mol, pos_mask=pos_mask)
        return data



class qm9_confs(geom_confs):
    def __init__(self, root, split_path, mode, transform=None, pre_transform=None, max_confs=10):
        super(qm9_confs, self).__init__(root, split_path, mode, transform, pre_transform, max_confs)
        self.types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}


class drugs_confs(geom_confs):
    def __init__(self, root, split_path, mode, transform=None, pre_transform=None, max_confs=10):
        super(drugs_confs, self).__init__(root, split_path, mode, transform, pre_transform, max_confs)
        self.types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
                      'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
                      'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
                      'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}


def construct_loader(args, modes=('train', 'val')):

    if isinstance(modes, str):
        modes = [modes]

    loaders = []
    for mode in modes:
        if args.dataset == 'qm9':
            dataset = qm9_confs(args.data_dir, args.split_path, mode, max_confs=args.n_true_confs)
        elif args.dataset == 'drugs':
            dataset = drugs_confs(args.data_dir, args.split_path, mode, max_confs=args.n_true_confs)
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False if mode == 'test' else True,
                            num_workers=args.num_workers,
                            pin_memory=False)
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders


bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
qm9_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
drugs_types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
               'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
               'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
               'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}


def featurize_mol_from_smiles(smiles, dataset='qm9'):

    if dataset == 'qm9':
        types = qm9_types
    elif dataset == 'drugs':
        types = drugs_types

    # filter fragments
    if '.' in smiles:
        return None

    # filter mols rdkit can't intrinsically handle
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.AddHs(mol)
    else:
        return None
    N = mol.GetNumAtoms()

    # filter out mols model can't make predictions for
    if not mol.HasSubstructMatch(dihedral_pattern):
        return None
    if N < 4:
        return None

    type_idx = []
    atomic_number = []
    atom_features = []
    chiral_tag = []
    neighbor_dict = {}
    ring = mol.GetRingInfo()
    for i, atom in enumerate(mol.GetAtoms()):
        type_idx.append(types[atom.GetSymbol()])
        n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(n_ids) > 1:
            neighbor_dict[i] = torch.tensor(n_ids)
        chiral_tag.append(chirality[atom.GetChiralTag()])
        atomic_number.append(atom.GetAtomicNum())
        atom_features.extend([atom.GetAtomicNum(),
                              1 if atom.GetIsAromatic() else 0])
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2]))
        atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
        atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                              int(ring.IsAtomInRingOfSize(i, 4)),
                              int(ring.IsAtomInRingOfSize(i, 5)),
                              int(ring.IsAtomInRingOfSize(i, 6)),
                              int(ring.IsAtomInRingOfSize(i, 7)),
                              int(ring.IsAtomInRingOfSize(i, 8))])
        atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))

    z = torch.tensor(atomic_number, dtype=torch.long)
    chiral_tag = torch.tensor(chiral_tag, dtype=torch.float)

    row, col, edge_type, bond_features = [], [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]
        bt = tuple(
            sorted([bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()])), bond.GetBondTypeAsDouble()
        bond_features += 2 * [int(bond.IsInRing()),
                              int(bond.GetIsConjugated()),
                              int(bond.GetIsAromatic())]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)
    # bond_features = torch.tensor(bond_features, dtype=torch.float).view(len(bond_type), -1)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    # edge_attr = torch.cat([edge_attr[perm], bond_features], dim=-1)
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()

    x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor(atom_features).view(N, -1)
    x = torch.cat([x1.to(torch.float), x2], dim=-1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, neighbors=neighbor_dict, chiral_tag=chiral_tag,
                name=smiles)
    data.edge_index_dihedral_pairs = get_dihedral_pairs(data.edge_index, data=data)

    return data
