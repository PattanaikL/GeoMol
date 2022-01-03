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
from rdkit.Chem.rdmolops import RemoveHs # We do not remove Hs to show add-hydgeon capability


def evaluate_conf(mols_ref_gen, useFF, threshold, removeH, maxmatches):
    smi, mols_ref, mols_gen = mols_ref_gen
    try:
        if removeH:
            rmsd_confusion_mat = get_rmsd_confusion_matrix_woh(mols_ref, mols_gen, useFF=useFF, maxmatches=maxmatches)
        else:
            rmsd_confusion_mat = get_rmsd_confusion_matrix(mols_ref, mols_gen, useFF=useFF, maxmatches=maxmatches)
        rmsd_ref_min_recall = rmsd_confusion_mat.min(-1)
        rmsd_ref_min_precision = rmsd_confusion_mat.min(0)
        return (
            (rmsd_ref_min_recall <= threshold).mean(),
            rmsd_ref_min_recall.mean(),
            (rmsd_ref_min_precision <= threshold).mean(),
            rmsd_ref_min_precision.mean(),
        )

    except RuntimeError:
        print(smi)
        return 0, 5, 0, 5


def group_conf_rmsd(mols_ref_gen, useFF, removeH, maxmatches):
    smi, _, mols_gen = mols_ref_gen
    group_rmsd = []
    # try:
    if not removeH:
        for i in range(len(mols_gen) - 1):
            for j in range(i + 1, len(mols_gen)):
                group_rmsd.append(MA.GetBestRMS(mols_gen[i], mols_gen[j], maxMatches=maxmatches))
    else:
        for i in range(len(mols_gen) - 1):
            for j in range(i + 1, len(mols_gen)):
                group_rmsd.append(MA.GetBestRMS(RemoveHs(mols_gen[i]), RemoveHs(mols_gen[j]), maxMatches=maxmatches))
    return sum(group_rmsd) / len(group_rmsd)

def get_rmsd_confusion_matrix(mols_ref, mols_gen, useFF, maxmatches):

    num_conf_ref = len(mols_ref)
    num_conf_gen = len(mols_gen)
    rmsd_confusion_mat = -1 * np.ones([num_conf_ref, num_conf_gen], dtype=np.float64)

    for i, conf_ref in enumerate(mols_ref):
        for j, conf_gen in enumerate(mols_gen):
            rmsd_confusion_mat[i, j] = MA.GetBestRMS(
                conf_gen, conf_ref, maxMatches=maxmatches
            )  # maxMatches is important for time
    return rmsd_confusion_mat

def get_rmsd_confusion_matrix_woh(mols_ref, mols_gen, useFF, maxmatches):

    num_conf_ref = len(mols_ref)
    num_conf_gen = len(mols_gen)
    rmsd_confusion_mat = -1 * np.ones([num_conf_ref, num_conf_gen], dtype=np.float64)

    for i, conf_ref in enumerate(mols_ref):
        for j, conf_gen in enumerate(mols_gen):
            rmsd_confusion_mat[i, j] = MA.GetBestRMS(
                RemoveHs(conf_gen), RemoveHs(conf_ref), maxMatches=maxmatches
            )  # maxMatches is important for time
    return rmsd_confusion_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get GeoMol results")
    parser.add_argument(
        "--ref",
        default="test_ref.pickle",
        type=PosixPath,
        help="absolute path of referenced confs' pickle",
    )
    parser.add_argument(
        "--file", type=str, help="name of tested pickle file",
    )
    parser.add_argument(
        "--testpath", type=PosixPath, help="path of testing, which pkl files exist"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="rmsd threshould, default to be"
    )
    parser.add_argument(
        "--FF",
        action="store_true",
        default=False,
        help="choose to use FF optimization from RDkit",
    )
    parser.add_argument(
        "--core", type=int, default=2, help="cpu cores used for computation"
    )
    parser.add_argument("--no", type=int, help="analyze the Nth molecule")
    parser.add_argument(
        "--removeH",
        action="store_true",
        default=False,
        help="choose to remove hydrogens for better RMSD",
    )
    parser.add_argument(
        "--rdkerrtxt",
        type=str, help="name of rdkit error smiles .txt file",
        default="rdkit_err_smiles.txt"
    )
    parser.add_argument(
        "--geoerrtxt",
        type=str, help="name of GeoMol error smiles .txt file",
        default="GeoMol_err_smiles.txt"
    )
    parser.add_argument(
        "--maxmatches",
        type=int, help="maxmatches for rdkit align",
        default=100
    )
    parser.add_argument(
        "--datatype",
        type=str, help="['train']",
        default=100
    )

    args = parser.parse_args()

    testfile = args.testpath / args.file
    ref_file = args.testpath / args.ref
    dirty_rdk_smis = args.testpath / args.rdkerrtxt
    dirty_GM_smis = args.testpath / args.geoerrtxt

    with open(ref_file, "rb") as f:
        data_ref = pickle.load(f)
    # dirty_smi_list = dirty_rdk_smis.read_text().split("\n") + dirty_GM_smis.read_text().split("\n") + ["CO[C@H]1[C@H]2CN3C[C@@H]1[C@@H]23"] + ['C[C@@H](O)[C@@H]1[C@H](O)[C@@H]2N[C@@H]21'] + ["'C[C@H]1[C@@H](C=O)N1C'"]
    dirty_smi_list = dirty_rdk_smis.read_text().split("\n") + dirty_GM_smis.read_text().split("\n")
    pklname = testfile.name[:-7]
    print(f"Dealing with {pklname}")

    with open(testfile, "rb") as f:
        data_gen = pickle.load(f)

    covs_R, covs_P = [], []
    mats_R, mats_P = [], []
    group_rmsd = []

    # data_gen_ref = [ (smi, mols_ref, data_gen[smi]) for smi,mols_ref in data_ref.items() if not smi in dirty_smi_list]
    data_gen_ref = [
        (smi, mols_ref, data_gen[smi])
        # (smi, [mols_ref], data_gen[smi]) # For single
        for smi, mols_ref in data_ref.items()
        if not smi in dirty_smi_list
    ]  # for platinum, where each mol has one conf


    data_gen_ref = [data_gen_ref[args.no - 1]] if args.no else data_gen_ref
    func1 = partial(
        evaluate_conf, useFF=args.FF, threshold=args.threshold, removeH=args.removeH, maxmatches=args.maxmatches
    )  # just for imap to take only one positional arguments

    func2 = partial(
        group_conf_rmsd, useFF=args.FF, removeH=args.removeH, maxmatches=args.maxmatches
    )

    with Pool(args.core) as pool:
        for result in tqdm(pool.imap(func1, data_gen_ref), total=len(data_gen_ref)):

            covs_R.append(result[0])
            mats_R.append(result[1])
            covs_P.append(result[2])
            mats_P.append(result[3])

        covs_R, mats_R, covs_P, mats_P = map(np.array, [covs_R, mats_R, covs_P, mats_P])
        
        for result in tqdm(pool.imap(func2, data_gen_ref), total=len(data_gen_ref)):
            group_rmsd.append(result)
        group_rmsd = np.array(group_rmsd)

    # # pool.close()
    # # pool.join()
    # # pool = multiprocessing.Pool(args.core)

    # for i, mol_gen_ref in tqdm(enumerate(data_gen_ref), total=len(data_gen_ref)):
        # group_rmsd.append(group_conf_rmsd(mol_gen_ref, useFF=args.FF, removeH=args.removeH))
    # group_rmsd = np.array(group_rmsd)


    # Write summary csv
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(['NumRotatableBonds'])
    numrots = [ calculator.CalcDescriptors(mol[0])[0] for _, mol in data_ref.items() ]
    # numrots = [ calculator.CalcDescriptors(mol)[0] for smi, mol in data_ref.items() if smi not in dirty_smi_list] # For single
    smis = list(data_ref.keys())
    indexes = [smis.index(smi) for smi in smis if smi not in dirty_smi_list]
    num_cov_mat_dict = {
        # 'No.': list(range(1,len(data_ref)+1)),
        'No.': indexes,
        'num_rotatable': numrots,
        'cov_R': covs_R,
        'mat_R': mats_R,
        'cov_P': covs_P,
        'mat_P': mats_P,
    }
    df = pd.DataFrame(num_cov_mat_dict)
################################################################################
    if not args.removeH:
        np.save(args.testpath / f"{pklname}-COV_R-th{args.threshold}-maxm{args.maxmatches}.npy", covs_R)
        np.save(args.testpath / f"{pklname}-MAT_R-th{args.threshold}-maxm{args.maxmatches}.npy", mats_R)
        np.save(args.testpath / f"{pklname}-COV_P-th{args.threshold}-maxm{args.maxmatches}.npy", covs_P)
        np.save(args.testpath / f"{pklname}-MAT_P-th{args.threshold}-maxm{args.maxmatches}.npy", mats_P)
        np.save(args.testpath / f"{pklname}-ingroup-rmsd-maxm{args.maxmatches}.npy", group_rmsd)

        print("Now the threshold is:", args.threshold)
        print("Now max matches for RMSD is:", args.maxmatches)
        print(
            "Coverage Mean (Recall): %.4f | Coverage Median (Recall): %.4f | Match Mean (Recall): %.4f | Match Median (Recall): %.4f"
            % (covs_R.mean(), np.median(covs_R), mats_R.mean(), np.median(mats_R))
        )
        print(
            "Coverage Mean (Precision): %.4f | Coverage Median (Precision): %.4f | Match Mean (Precision): %.4f | Match Median (Precision): %.4f"
            % (covs_P.mean(), np.median(covs_P), mats_P.mean(), np.median(mats_P))
        )

        outfile = args.testpath / f"{pklname}-th{args.threshold}-maxm{args.maxmatches}.csv"
        df.to_csv(outfile, index=False)
    
    else:
        np.save(args.testpath / f"{pklname}-COV_R-th{args.threshold}-woh-maxm{args.maxmatches}.npy", covs_R)
        np.save(args.testpath / f"{pklname}-MAT_R-th{args.threshold}-woh-maxm{args.maxmatches}.npy", mats_R)
        np.save(args.testpath / f"{pklname}-COV_P-th{args.threshold}-woh-maxm{args.maxmatches}.npy", covs_P)
        np.save(args.testpath / f"{pklname}-MAT_P-th{args.threshold}-woh-maxm{args.maxmatches}.npy", mats_P)
        np.save(args.testpath / f"{pklname}-ingroup-rmsd-woh-maxm{args.maxmatches}.npy", group_rmsd)

        print("Now the threshold is:", args.threshold)
        print("Now max matches for RMSD is:", args.maxmatches)
        print("Note now removeH has been ignited")
        print(
            "Coverage Mean (Recall): %.4f | Coverage Median (Recall): %.4f | Match Mean (Recall): %.4f | Match Median (Recall): %.4f"
            % (covs_R.mean(), np.median(covs_R), mats_R.mean(), np.median(mats_R))
        )
        print(
            "Coverage Mean (Precision): %.4f | Coverage Median (Precision): %.4f | Match Mean (Precision): %.4f | Match Median (Precision): %.4f"
            % (covs_P.mean(), np.median(covs_P), mats_P.mean(), np.median(mats_P))
        )

        outfile = args.testpath / f"{pklname}-th{args.threshold}-removeH-maxm{args.maxmatches}.csv"
        df.to_csv(outfile, index=False)