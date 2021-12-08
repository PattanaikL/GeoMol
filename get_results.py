"""
For the Cov and MAT results of GeoMol
Refer to codes from GeoMol and ConfGF
"""
import pickle
import numpy as np
import argparse
from pathlib import Path, PosixPath
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule # from ConfGF
from rdkit.Chem import rdMolAlign as MA
# from rdkit.Chem.rdmolops import RemoveHs # We do not remove Hs to show add-hydgeon capability
import multiprocessing
from functools import partial
import tqdm


"""scripts from ConfGF 
def GetBestRMSD(probe, ref):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)
    rmsd = MA.GetBestRMS(probe, ref)
    return rmsd
    
def get_rmsd_confusion_matrix(data: Data, useFF=False):
    data.pos_ref = data.pos_ref.view(-1, data.num_nodes, 3)
    data.pos_gen = data.pos_gen.view(-1, data.num_nodes, 3)
    num_gen = data.pos_gen.size(0)
    num_ref = data.pos_ref.size(0)

    assert num_gen == data.num_pos_gen.item()
    assert num_ref == data.num_pos_ref.item()

    rmsd_confusion_mat = -1 * np.ones([num_ref, num_gen],dtype=np.float)
    
    # for i in range(num_gen):
    #     gen_mol = utils.set_rdmol_positions(data.rdmol, data.pos_gen[i])
    #     if useFF:
    #         #print('Applying FF on generated molecules...')
    #         MMFFOptimizeMolecule(gen_mol)
    #     for j in range(num_ref):
    #         ref_mol = utils.set_rdmol_positions(data.rdmol, data.pos_ref[j])
            
    rmsd_confusion_mat[j,i] = GetBestRMSD(gen_mol, ref_mol)

    return rmsd_confusion_mat

def evaluate_conf(data, useFF=False, threshold=0.5):
    rmsd_confusion_mat = get_rmsd_confusion_matrix(gendata, refdata, useFF=useFF)
    rmsd_ref_min = rmsd_confusion_mat.min(-1)
    return (rmsd_ref_min<=threshold).mean(), rmsd_ref_min.mean()
"""

def evaluate_conf(mols_ref_gen, useFF=False, threshold=0.5):
    mols_ref, mols_gen = mols_ref_gen
    rmsd_confusion_mat = get_rmsd_confusion_matrix(mols_ref, mols_gen, useFF=useFF)
    rmsd_ref_min = rmsd_confusion_mat.min(-1)
    return (rmsd_ref_min<=threshold).mean(), rmsd_ref_min.mean()

# def get_rmsd_confusion_matrix(mols_gen, mols_ref, useFF=False):
def get_rmsd_confusion_matrix(mols_ref, mols_gen, useFF=False):

    num_conf_ref = len(mols_ref)
    num_conf_gen = len(mols_gen)
    rmsd_confusion_mat = -1 * np.ones([num_conf_ref, num_conf_gen],dtype=np.float64)

    for i, conf_ref in enumerate(mols_ref):
        for j, conf_gen in enumerate(mols_gen):
            rmsd_confusion_mat[i,j] = MA.GetBestRMS(conf_gen, conf_ref)
    return rmsd_confusion_mat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get GeoMol results")
    parser.add_argument("--ref", type=PosixPath, help="absolute path of referenced confs' pickle", required=True)
    parser.add_argument("--gen", type=PosixPath, help="absolute path of GeoMol generated confs' pickle")
    parser.add_argument("--rdkit", type=PosixPath, help="absolute path of rdkit generated confs' pickle")
    parser.add_argument("--threshold", type=float, default=0.5, help="rmsd threshould, default to be")
    parser.add_argument("--FF", action="store_true", default=False, help="choose to use FF optimization from RDkit")
    parser.add_argument("--core", type=int, default=2, help="cpu cores used for computation")
    args = parser.parse_args()

    testfile = args.gen if args.gen else args.rdkit

    with open(testfile, 'rb') as f:
        data_gen = pickle.load(f)
    with open(args.ref, 'rb') as f:
        data_ref = pickle.load(f)

    covs = []
    mats = []

    # dirty_smi_list = ["[H]O[C@]1(C([H])([H])C([H])([H])[H])[C@]2([H])C([H])([H])[C@@]1([H])C2([H])[H]","[H]C([H])([H])C(=O)[C@@]12N3[C@@]([H])(C1([H])[H])[C@@]3([H])C2([H])[H]","[H]N1[C@@]23C([H])([H])O[C@@]2([H])[C@]1([H])[C@]1([H])N([H])[C@]31[H]","[H]OC([H])([H])[C@]12C([H])([H])[C@]([H])(C1([H])[H])[C@]2([H])C([H])([H])[H]","[H]C1([H])N2C(C#N)=N[C@]([H])(C([H])([H])[H])[C@@]21[H]","[H]O[C@@]1([H])C([H])([H])[C@@]23O[C@@]4([H])[C@@]12O[C@@]34[H]","[H]C1([H])O[C@@]1([H])[C@]1([H])[C@]2([H])N3C([H])([H])[C@]3([H])[C@]12[H]","[H]O[C@]1(C([H])([H])[H])[C@@]2([H])C([H])([H])C(C([H])([H])[H])(C([H])([H])[H])[C@]21[H]","[H]C([H])([H])[C@@]1([H])[C@]2([H])[C@]3([H])C([H])([H])[C@@](C([H])([H])[H])(C3([H])[H])[C@]12[H]",]# for rdkit
    dirty_smi_list = ["CC[C@]1(O)[C@H]2C[C@@H]1C2","CC(=O)[C@@]12C[C@H]3[C@@H](C1)N32","C1O[C@H]2[C@H]3N[C@@]12[C@H]1N[C@@H]31","C[C@H]1[C@H]2C[C@]1(CO)C2","C[C@H]1N=C(C#N)N2C[C@@H]12","O[C@H]1C[C@]23O[C@@H]4[C@H]2O[C@@]143","C1O[C@H]1[C@H]1[C@H]2[C@H]3CN3[C@@H]12","CC1(C)C[C@H]2[C@@H]1[C@]2(C)O","C[C@H]1[C@@H]2[C@H]3C[C@](C)(C3)[C@H]12",]
    data_gen_ref = [ (mols_ref, data_gen[smi]) for smi,mols_ref in data_ref.items() if not smi in dirty_smi_list]
    
    # data_gen_ref = [ (mols_ref, data_gen[smi]) for smi,mols_ref in data_ref.items() ]

    pool = multiprocessing.Pool(args.core)
    func = partial(evaluate_conf, useFF=args.FF, threshold=args.threshold) # just for imap to take only one positional arguments
    # for result in tqdm(pool.imap(func, data_gen_ref), total=len(data_gen_ref)):
    for result in pool.imap(func, data_gen_ref):
        covs.append(result[0])
        mats.append(result[1])
    covs = np.array(covs)
    mats = np.array(mats)

    # for result in pool.imap(func,data_gen_ref):
        # print(result)
    pool.close()
    pool.join()
    print(covs)
    print(mats)


    print('Coverage Mean: %.4f | Coverage Median: %.4f | Match Mean: %.4f | Match Median: %.4f' % \
                        (covs.mean(), np.median(covs), mats.mean(), np.median(mats)))
    