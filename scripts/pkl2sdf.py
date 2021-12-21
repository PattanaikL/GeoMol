import pickle
from rdkit import Chem
from pathlib import Path, PosixPath
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=PosixPath, help="absolute path for sdf generation", required=True)
    parser.add_argument("--pkl_file", type=str, help="absolute path of pickle files", required=True)
    args = parser.parse_args()
    # pkl_path = Path(os.environ['HOME']) / "git-repo/AI-CONF/datasets/GeoMol/test" / "drugs-plati"
    pkl_path = args.pkl_path
    pkl_file = pkl_path / args.pkl_file
    
    sdf_dir = pkl_path / "split_mols" / pkl_file.name.split(".")[0]
    if not sdf_dir.exists():
        # os.system(f"mkdir -p {sdf_dir}")
        os.makedirs(sdf_dir)
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    i = 1
    for smi, mols in data.items():
        writer = Chem.SDWriter(str(sdf_dir / (str(i).zfill(4) + ".sdf")))
        if isinstance(mols, list):
            for mol in mols:
                mol.SetProp("_Name" , f"{pkl_file.name.split('.')[0]}_{str(i).zfill(4)}")
                # mol.SetProp("_SourceID","%s" % i)
                mol.SetProp("_SMILES", "%s" % smi)    
                writer.write(mol)
        else:
            mols.SetProp("_Name" , f"{pkl_file.name.split('.')[0]}_{str(i).zfill(4)}")
            # mol.SetProp("_SourceID","%s" % i)
            mols.SetProp("_SMILES", "%s" % smi)    
            writer.write(mols)
            
        writer.close()
        i+=1