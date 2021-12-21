from rdkit import Chem
import pandas as pd
from pathlib import Path, PosixPath
import pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workpath", type=PosixPath, help="absolute path for pkl generation", required=True)
    parser.add_argument("--sdf", type=str, help="absolute path of sdf file", required=True)
    args = parser.parse_args()

    # workpath = Path("/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-plati")
    workpath = args.workpath
    sdffile = args.sdf

    # mols = Chem.SDMolSupplier("platinum_diverse_dataset_2017_01.sdf", removeHs= False)
    # molswoh = Chem.SDMolSupplier("platinum_diverse_dataset_2017_01.sdf")

    mols = Chem.SDMolSupplier(sdffile, removeHs= False)
    molswoh = Chem.SDMolSupplier(sdffile)

    smis = list(map(Chem.MolToSmiles, mols))
    smis_woh = list(map(Chem.MolToSmiles, molswoh))
    molswoh_addh = list(map(Chem.AddHs, molswoh))
    
    maxconfs = 25
    # df = pd.DataFrame({'smiles':smis, 'n_conformers':maxconfs})
    # df.to_csv("test_smiles.csv")
    
    df = pd.DataFrame({'smiles':smis_woh, 'n_conformers':maxconfs})
    df.to_csv(workpath / "test_smiles.csv", index=False)
    
    
    plati_smis_mols_dict = dict(zip(smis,mols))
    plati_smis_woh_mols_dict = dict(zip(smis_woh, mols))
    plati_smis_mols_addh_dict = dict(zip(smis, molswoh_addh))
    plati_smis_woh_mols_addh_dict = dict(zip(smis_woh, molswoh_addh)) # Dedicated for smiles -> addh step in featurization
    
    with open(workpath / "test_ref.pickle", 'wb') as f:
        # pickle.dump(plati_smis_woh_mols_addh_dict, f)
        pickle.dump(plati_smis_woh_mols_dict, f)