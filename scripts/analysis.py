# %%
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import argparse
from pathlib import Path, PosixPath
from collections import Counter
from rdkit.ML.Descriptors import MoleculeDescriptors

"""
Usage:
python analysis.py --testpath /pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-split0/12-11-15-37 \
--covnpy test_GeoMol-COV_P-th0.5.npy \
--matnpy test_GeoMol-MAT_P-th0.5.npy \
--outprefix test_GeoMol-th0.5
    
"""
if __name__ == "__main__":

    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(['NumRotatableBonds'])
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--testpath", type=PosixPath, help="absolute path of all test files")
    parser.add_argument("--ref", type=str, default="test_ref.pickle", help="name of ref pickle")
    parser.add_argument("--covnpy", type=str, help="name of cov .npy file", required=True)
    parser.add_argument("--matnpy", type=str, help="name of mat .npy file", required=True)
    parser.add_argument("--outprefix", type=str, help="prefix of output file", required=True)
    args = parser.parse_args()

    with open(args.testpath / args.ref, 'rb') as f:
        refdata = pickle.load(f)
    numrots = [ calculator.CalcDescriptors(mol[0])[0] for _, mol in refdata.items() ]

    covfile = args.testpath / args.covnpy
    matfile = args.testpath / args.matnpy

    covs = list(np.load(covfile))
    mats = list(np.load(matfile))

    num_cov_mat_dict = {
        'No.': list(range(1,len(refdata)+1)),
        'num_rotatable': numrots,
        'cov': covs,
        'mats': mats
    }
    df = pd.DataFrame(num_cov_mat_dict)
    # outfile = args.testpath / (args.outprefix + ".csv")
    # df.to_csv(outfile, index=False)

# %%
# """
    # print(np.where(covs==1.))
    # print(len(np.where(covs==1.)[0]))

    recounted = Counter(df['num_rotatable'].values)
    # plt.plot()
    df_cov1 = df[df['cov'] == 1.0]
    # 
    # plt.hist(x=df['num_rotatable'].values,bins = 30)
    # plt.hist(x=recounted ,bins = 30)

    # df['num_rotatable'].values
    # recounted
    sns.distplot(df['num_rotatable'].values, bins=10)
    # sns.histplot(df_cov1['num_rotatable'],bins=10, edgecolor="black")

    plt.hist(x=df_cov1['num_rotatable'],bins=10, edgecolor="black")


    # import numpy as np

    datapath = "/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-plati/test_GeoMol_20-Geo-rmsd.npy"
    data = np.load(datapath)

    data.mean()
# """