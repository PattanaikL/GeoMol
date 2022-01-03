"""
For the Cov and MAT results of GeoMol
Refer to codes from GeoMol and ConfGF
"""
import numpy as np
import pandas as pd
import argparse
import pickle

from pathlib import Path, PosixPath
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule  # from ConfGF
from rdkit.Chem import rdMolAlign as MA
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcNumRings
from rdkit.Chem.rdmolops import RemoveHs  # We do not remove Hs to show add-hydgeon capability

def evaluate_conf_multi(smi_mols_1_2, useFF, threshold, removeH, maxmatches):
    """
    :param smi_mols_1_2: (smi, mols_1, mols_2)
    :param useFF: whether to use MMFF94s for optimization
    :param threshold: threshold for filtering distant confs from RDKit and GeoMol
    :param removeH: whether to use RemoveHs to delete hydrogens in generated confs
    :param maxmatches: max matches used by GetBestRMS 
    :return smi, indexes
        smi: smiles (type:str)
        indexes: list of indexes for confs in mols_1 while not in mols_2 (type:list of int)
    """
    smi, mols_1, mols_2 = smi_mols_1_2
    try:
        if removeH:
            rmsd_confusion_mat = get_rmsd_confusion_matrix_woh(
                mols_1, mols_2, useFF=useFF, maxmatches=maxmatches
            )
        else:
            rmsd_confusion_mat = get_rmsd_confusion_matrix(
                mols_1, mols_2, useFF=useFF, maxmatches=maxmatches
            )
        rmsd_ref_min_recall = rmsd_confusion_mat.min(-1)
        # rmsd_ref_min_precision = rmsd_confusion_mat.min(0)
        # for threshold in [1.0, 2.0, 5.0]:
            # indexes = np.where(rmsd_ref_min_recall > threshold)[0]
            # yield smi, indexes, len(indexes)
        indexes1 = np.where(rmsd_ref_min_recall > 1.0)[0]
        indexes2 = np.where(rmsd_ref_min_recall > 2.0)[0]
        indexes5 = np.where(rmsd_ref_min_recall > 5.0)[0]

        return [[smi, indexes1, len(indexes1)],[smi, indexes2, len(indexes2)],[smi, indexes5, len(indexes5)]]

    except RuntimeError:
        print("Evaluation Failed!!!")
        print("Please have notion with smi:", smi)
        return

def evaluate_conf(smi_mols_1_2, useFF, threshold, removeH, maxmatches):
    """
    :param smi_mols_1_2: (smi, mols_1, mols_2)
    :param useFF: whether to use MMFF94s for optimization
    :param threshold: threshold for filtering distant confs from RDKit and GeoMol
    :param removeH: whether to use RemoveHs to delete hydrogens in generated confs
    :param maxmatches: max matches used by GetBestRMS 
    :return smi, indexes
        smi: smiles (type:str)
        indexes: list of indexes for confs in mols_1 while not in mols_2 (type:list of int)
    """
    smi, mols_1, mols_2 = smi_mols_1_2
    try:
        if removeH:
            rmsd_confusion_mat = get_rmsd_confusion_matrix_woh(
                mols_1, mols_2, useFF=useFF, maxmatches=maxmatches
            )
        else:
            rmsd_confusion_mat = get_rmsd_confusion_matrix(
                mols_1, mols_2, useFF=useFF, maxmatches=maxmatches
            )
        rmsd_ref_min_recall = rmsd_confusion_mat.min(-1)
        # rmsd_ref_min_precision = rmsd_confusion_mat.min(0)
        indexes = np.where(rmsd_ref_min_recall > threshold)
        return smi, indexes, len(indexes)

    except RuntimeError:
        print("Evaluation Failed!!!")
        print("Please have notion with smi:", smi)
        return


def group_conf_rmsd(smi_mols, useFF, removeH, maxmatches):
    """
    :param smi_mols: dict {smi:mols}
    :param useFF: whether to use MMFF94s for optimization
    :param maxmatches: maxmatches used by GetBestRMS
    :return mean in-group RMSD (type:float)
    """
    smi, mols = smi_mols
    group_rmsd = []
    try:
        if not removeH:
            for i in range(len(mols) - 1):
                for j in range(i + 1, len(mols)):
                    group_rmsd.append(
                        MA.GetBestRMS(mols[i], mols[j], maxMatches=maxmatches)
                    )
        else:
            for i in range(len(mols) - 1):
                for j in range(i + 1, len(mols)):
                    group_rmsd.append(
                        MA.GetBestRMS(
                            RemoveHs(mols[i]), RemoveHs(mols[j]), maxMatches=maxmatches
                        )
                    )
    except RuntimeError:
        print("In Group RMSD calc Failed!!!")
        print("Please have notion with smi:", smi)
    return sum(group_rmsd) / len(group_rmsd)


def get_rmsd_confusion_matrix(mols_ref, mols_gen, useFF, maxmatches):

    num_conf_ref = len(mols_ref)
    num_conf_gen = len(mols_gen)
    rmsd_confusion_mat = -1 * np.ones([num_conf_ref, num_conf_gen], dtype=np.float64)

    for i, conf_ref in enumerate(mols_ref):
        for j, conf_gen in enumerate(mols_gen):
            rmsd_confusion_mat[i, j] = MA.GetBestRMS(
                conf_ref, conf_gen, maxMatches=maxmatches
            )  # maxMatches is important for time
    return rmsd_confusion_mat


def get_rmsd_confusion_matrix_woh(mols_ref, mols_gen, useFF, maxmatches):

    num_conf_ref = len(mols_ref)
    num_conf_gen = len(mols_gen)
    rmsd_confusion_mat = -1 * np.ones([num_conf_ref, num_conf_gen], dtype=np.float64)

    for i, conf_ref in enumerate(mols_ref):
        for j, conf_gen in enumerate(mols_gen):
            rmsd_confusion_mat[i, j] = MA.GetBestRMS(
                RemoveHs(conf_ref), RemoveHs(conf_gen), maxMatches=maxmatches
            )  # maxMatches is important for time
    return rmsd_confusion_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare confs generated by different methods")
    parser.add_argument("--testpath", type=PosixPath, help="absolute path for tested pickle files")
    parser.add_argument("--mols1", type=str, default="test_GeoMol.pickle", help="name of mols1 pickle, for ref")
    parser.add_argument("--mols2", type=str, default="test_rdkit.pickle", help="name of mols2 pickle, for comparison")
    parser.add_argument("--threshold", type=float, default=2.0, help="rmsd threshold, default to be 2.0")
    parser.add_argument("--maxmatches", type=int, help="maxmatches for rdkit align", default=100)
    parser.add_argument("--core", type=int, default=2, help="cpu cores used for computation")
    parser.add_argument(
        "--FF",
        action="store_true",
        default=False,
        help="choose to use FF optimization from RDkit",)
    parser.add_argument(
        "--removeH",
        action="store_true",
        default=False,
        help="choose to remove hydrogens for better RMSD",)
    parser.add_argument(
        "--rdkerrtxt",
        type=str,
        help="name of rdkit error smiles .txt file",
        default="rdkit_err_smiles.txt",)
    parser.add_argument(
        "--geoerrtxt",
        type=str,
        help="name of GeoMol error smiles .txt file",
        default="GeoMol_err_smiles.txt",)
    args = parser.parse_args()

    mols1 = args.testpath / args.mols1
    mols2 = args.testpath / args.mols2
    dirty_rdk_smis = args.testpath / args.rdkerrtxt
    dirty_GM_smis = args.testpath / args.geoerrtxt

    dirty_smi_list = dirty_rdk_smis.read_text().split("\n") + dirty_GM_smis.read_text().split("\n")
    # print(dirty_smi_list)
# '''
    with open(mols1, "rb") as f:
        data_mols1 = pickle.load(f)
    with open(mols2, "rb") as f:
        data_mols2 = pickle.load(f)
    smis = list(data_mols1.keys())
    smi_index = [smis.index(smi) + 1 for smi in smis if smi not in dirty_smi_list]

    data_smi_mols_1_2 = [
        (smi, mols_1, data_mols2[smi])
        # (smi, [mols_1], data_mols2[smi])  # For single
        for smi, mols_1 in data_mols1.items()
        if not smi in dirty_smi_list
    ]
    data_mols1 = [(smi, mols_1) for smi, mols_1, _ in data_smi_mols_1_2]
    data_mols2 = [(smi, mols_2) for smi, _, mols_2 in data_smi_mols_1_2]


    func1 = partial(
        evaluate_conf,
        useFF=args.FF,
        threshold=args.threshold,
        removeH=args.removeH,
        maxmatches=args.maxmatches,
    )  # just for imap to take only one positional arguments

    func3 = partial(
        evaluate_conf_multi,
        useFF=args.FF,
        threshold=args.threshold,
        removeH=args.removeH,
        maxmatches=args.maxmatches,
    )  # just for imap to take only one positional arguments
    func2 = partial(
        group_conf_rmsd, useFF=args.FF, removeH=args.removeH, maxmatches=args.maxmatches
    )

    smi_list = []
    indexes1_list, indexes2_list, indexes5_list = [], [], []
    index1_lens, index2_lens, index5_lens = [], [], []
    group_rmsd_1 = []
    group_rmsd_2 = []
    # """
    with Pool(args.core) as pool:
        # for result in tqdm(pool.imap(func1, data_smi_mols_1_2), total=len(data_smi_mols_1_2)):
            # if result:
                # smi_list.append(result[0])
                # indexes_list.append(result[1])
                # index_lens.append(result[2])

        for result in tqdm(pool.imap(func3, data_smi_mols_1_2), total=len(data_smi_mols_1_2)):

            if result:
                result = list(result)
                smi_list.append(result[0][0])
                indexes1_list.append(result[0][1])
                index1_lens.append(result[0][2])
                indexes2_list.append(result[1][1])
                index2_lens.append(result[1][2])
                indexes5_list.append(result[2][1])
                index5_lens.append(result[2][2])
        # for result in tqdm(pool.imap(func2, data_mols1.items()), total=len(data_mols1)):
        # for result in tqdm(pool.imap(func2, data_mols1), total=len(data_mols1)):
            # if result:
                # group_rmsd_1.append(result)
        # for result in tqdm(pool.imap(func2, data_mols2.items()), total=len(data_mols2)):
        # for result in tqdm(pool.imap(func2, data_mols2), total=len(data_mols2)):
            # if result:
                # group_rmsd_2.append(result)
# 
        # group_rmsd_1 = np.array(group_rmsd_1)
        # group_rmsd_2 = np.array(group_rmsd_2)
    # """
    # Write summary csv
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
        ["NumRotatableBonds"]
    )
    numrots1 = [
        # calculator.CalcDescriptors(mol)[0] # For single
        calculator.CalcDescriptors(mol[0])[0] 
        for smi, mol in data_mols2
        if smi not in dirty_smi_list
    ]
    numrots2 = [
        # CalcNumRotatableBonds(mol)[0] # For single
        CalcNumRotatableBonds(mol[0]) 
        for smi, mol in data_mols2
        if smi not in dirty_smi_list
    ]
    numrings = [
        # CalcNumRings(mol)[0] # For single
        CalcNumRings(mol[0])
        for smi, mol in data_mols2
        if smi not in dirty_smi_list
    ]
    numconfs_1 = [
        len(mol)
        for smi, mol in data_mols1
        if smi not in dirty_smi_list
    ]
    numconfs_2 = [
        len(mol)
        for smi, mol in data_mols2
        if smi not in dirty_smi_list
    ]

    # numrot_smi_indexes_group_rmsd_dict = {
        # "No.": smi_index,
        # "smi": smi_list,
        # "indexes": indexes_list,
        # "num_rotatable1": numrots1,
        # "num_rotatable2": numrots2,
        # "group_rmsd_1": group_rmsd_1,
        # "group_rmsd_2": group_rmsd_2
    # }

    numrot_smi_indexes_group_rmsd_dict = {
        "No.": smi_index,
        "smi": smi_list,
        "indexes_1": indexes1_list,
        "indexes_2": indexes2_list,
        "indexes_5": indexes5_list,
        "index_len1": index1_lens,
        "index_len2": index2_lens,
        "index_len5": index5_lens,
        "num_rotatable-ML": numrots1,
        "num_rotatable-Desc": numrots2,
        "num_rings": numrings,
        "numconfs_1": numconfs_1,
        "numconfs_2": numconfs_2,
        # "group_rmsd_1": group_rmsd_1,
        # "group_rmsd_2": group_rmsd_2
    }
    df = pd.DataFrame(numrot_smi_indexes_group_rmsd_dict)
    outname = mols1.stem + "-" + mols2.stem

    ################################################################################
    if not args.removeH:
        print("Now the threshold is:", args.threshold)
        print("Now max matches for RMSD is:", args.maxmatches)
        # print(
            # "Coverage Mean (Recall): %.4f | Coverage Median (Recall): %.4f | Match Mean (Recall): %.4f | Match Median (Recall): %.4f"
            # % (covs_R.mean(), np.median(covs_R), mats_R.mean(), np.median(mats_R))
        # )
        # print(
            # "Coverage Mean (Precision): %.4f | Coverage Median (Precision): %.4f | Match Mean (Precision): %.4f | Match Median (Precision): %.4f"
            # % (covs_P.mean(), np.median(covs_P), mats_P.mean(), np.median(mats_P))
        # )
# 
        outfile = (args.testpath / f"{outname}-th{args.threshold}-maxm{args.maxmatches}-sumamry.csv")
        df.to_csv(outfile, index=False)

    else:
        print("Now the threshold is:", args.threshold)
        print("Now max matches for RMSD is:", args.maxmatches)
        print("Note now removeH has been ignited")
        # print(
            # "Coverage Mean (Recall): %.4f | Coverage Median (Recall): %.4f | Match Mean (Recall): %.4f | Match Median (Recall): %.4f"
            # % (covs_R.mean(), np.median(covs_R), mats_R.mean(), np.median(mats_R))
        # )
        # print(
            # "Coverage Mean (Precision): %.4f | Coverage Median (Precision): %.4f | Match Mean (Precision): %.4f | Match Median (Precision): %.4f"
            # % (covs_P.mean(), np.median(covs_P), mats_P.mean(), np.median(mats_P))
        # )
# 
        outfile = (args.testpath / f"{outname}-th{args.threshold}-maxm{args.maxmatches}-removeH-sumamry.csv")
        df.to_csv(outfile, index=False)

# '''