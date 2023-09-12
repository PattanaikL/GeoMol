from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType, HybridizationType
from rdkit.Chem.rdchem import BondType as BT

import glob
import os.path as osp
import pickle
import random
from typing import Optional
from packaging import version

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.data import Batch, Data, DataLoader, Dataset
from torch_scatter import scatter

from geomol.utils import get_dihedral_pairs

tg_version_ge_2 = version.parse(tg.__version__) > version.parse('2.0.0')

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}
dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')

qm9_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
drugs_types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
               'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
               'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
               'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}
dataset_types = {'qm9': qm9_types, 'drugs': drugs_types}


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

    dataset = ''

    def __init__(self,
                 root,
                 split_path,
                 mode,
                 transform=None,
                 pre_transform=None,
                 max_confs=10):
        super().__init__(root, transform, pre_transform)

        self.root = root
        self.split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        self.split = np.load(split_path, allow_pickle=True)[self.split_idx]
        self.bonds = bonds

        self.dihedral_pairs = {}  # for memoization
        all_files = sorted(glob.glob(osp.join(self.root, '*.pickle')))
        self.pickle_files = [f for i, f in enumerate(all_files)
                             if i in self.split]
        self.max_confs = max_confs
        self.types = dataset_types[self.dataset]

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
        confs, name = mol_dic['conformers'], mol_dic["smiles"]
        random.shuffle(confs)  # shuffle confs

        # filter mols rdkit can't intrinsically handle
        try:
            canonical_smi = Chem.MolToSmiles(Chem.MolFromSmiles(name))
        except Exception:
            return None

        # skip conformers without dihedrals
        if _check_mol(confs[0]['rd_mol'], smiles=name) is None:
            return None

        n_atom = confs[0]['rd_mol'].GetNumAtoms()
        pos = torch.zeros([self.max_confs, n_atom, 3])
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
            except Exception:
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

        x, z, edge_index, edge_attr, neighbor_dict, chiral_tag \
            = _mol_to_features(correct_mol, self.dataset)

        data = Data(x=x, z=z, pos=[pos],
                    edge_index=edge_index, edge_attr=edge_attr,
                    neighbors=neighbor_dict,
                    chiral_tag=chiral_tag,
                    name=name, mol=correct_mol,
                    boltzmann_weight=conf['boltzmannweight'],
                    degeneracy=conf['degeneracy'],
                    pos_mask=pos_mask)
        return data


class qm9_confs(geom_confs):

    dataset = 'qm9'


class drugs_confs(geom_confs):

    dataset = 'drugs'


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


def smiles_to_mol(smiles: str,
                  check_mol: bool = True):
    """
    Convert a SMILES string to a RDKit molecule.
    """
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    except Exception:
        return None
    if check_mol:
        return _check_mol(mol, smiles=smiles)
    return mol


def _check_mol(mol,
               smiles=None):
    """
    Check if a molecule is valid.
    """
    # filter fragments
    if smiles is not None:
        if '.' in smiles:
            return None
    else:
        frags = Chem.rdmolops.GetMolFrags(mol,
                                          asMols=False)
        if len(frags) > 1:
            return None

    # filter out mols model can't make predictions for
    if mol.GetNumAtoms() < 4:
        return None
    if mol.GetNumBonds() < 4:
        # in Lucky' original implementation
        # this criteria is included in geom_confs.featurize_mol
        # but not included in featurize_mol_from_smiles
        # add it here anyway
        return None
    if not mol.HasSubstructMatch(dihedral_pattern):
        return None
    return mol


def _mol_to_features(mol,
                     dataset: str = 'qm9'):
    """
    Prepare necessary information for converting a RDKit mol object to a torch_geometry_data object.
    """
    types = dataset_types[dataset]

    type_idx = []
    atomic_number = []
    atom_features = []
    chiral_tag = []
    neighbor_dict = {}
    ring = mol.GetRingInfo()

    n_atom = mol.GetNumAtoms()
    # Atomic features
    for i, atom in enumerate(mol.GetAtoms()):
        type_idx.append(types[atom.GetSymbol()])
        if len(atom.GetNeighbors()) > 1:
            n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
            neighbor_dict[i] = torch.tensor(n_ids)
        chiral_tag.append(chirality[atom.GetChiralTag()])
        atomic_number.append(atom.GetAtomicNum())
        atom_features.extend([atom.GetAtomicNum(),
                              1 if atom.GetIsAromatic() else 0])
        atom_features.extend(one_k_encoding(
            atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2]))
        atom_features.extend(one_k_encoding(
            atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(
            atom.GetFormalCharge(), [-1, 0, 1]))
        atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                              int(ring.IsAtomInRingOfSize(i, 4)),
                              int(ring.IsAtomInRingOfSize(i, 5)),
                              int(ring.IsAtomInRingOfSize(i, 6)),
                              int(ring.IsAtomInRingOfSize(i, 7)),
                              int(ring.IsAtomInRingOfSize(i, 8))])
        atom_features.extend(one_k_encoding(
            int(ring.NumAtomRings(i)), [0, 1, 2, 3]))

    z = torch.tensor(atomic_number, dtype=torch.long)
    chiral_tag = torch.tensor(chiral_tag, dtype=torch.float)

    # Edge features
    row, col, edge_type, bond_features = [], [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]
        bt = tuple(sorted(
            [bond.GetBeginAtom().GetAtomicNum(),
             bond.GetEndAtom().GetAtomicNum()]
        )), bond.GetBondTypeAsDouble()
        bond_features += 2 * [int(bond.IsInRing()),
                              int(bond.GetIsConjugated()),
                              int(bond.GetIsAromatic())]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    perm = (edge_index[0] * n_atom + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=n_atom).tolist()

    x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor(atom_features).view(n_atom, -1)
    x = torch.cat([x1.to(torch.float), x2], dim=-1)

    return x, z, edge_index, edge_attr, neighbor_dict, chiral_tag


def featurize_mol(mol,
                  dataset: str = 'qm9',
                  smiles: Optional[str] = None,
                  name: str = ''):
    """
    Featurize a molecule.
    """
    mol = _check_mol(mol, smiles=smiles)
    name = smiles if (smiles and not name) else name

    if mol:
        x, _, edge_index, edge_attr, neighbor_dict, chiral_tag \
            = _mol_to_features(mol, dataset=dataset)
        data = Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    neighbors=neighbor_dict,
                    chiral_tag=chiral_tag,
                    name=name)
        data.edge_index_dihedral_pairs \
            = get_dihedral_pairs(data.edge_index,
                                 data=data)
        return data


def featurize_mol_from_smiles(smiles: str,
                              dataset='qm9'):
    """
    Featurize a molecule from a SMILES string.
    """
    mol = smiles_to_mol(smiles, check_mol=True)
    if mol:
        return featurize_mol(mol,
                             dataset=dataset,
                             name=smiles)


def from_data_list(data_list: list):
    """
    Creates a batch object from a list of data objects. This is useful for inference with an improvisational list of features from different molecules.
    This function is a wrapper for the torch_geometric function Batch.from_data_list
    with a special treatment for the neighbors attribute. If without the
    treatment, neighbors will be collapsed into a single dict and only have keys in the
    first elements, causing an error raised in "get_neighbor_ids".

    It has only been tested and applied for torch_geometric over version 2.0.0.
    """
    if tg_version_ge_2:
        batch_data = Batch.from_data_list(data_list,
                                          exclude_keys=['neighbors'])
        batch_data.neighbors = [d.neighbors for d in data_list]
    else:
        batch_data = Batch.from_data_list(data_list)
    return batch_data
