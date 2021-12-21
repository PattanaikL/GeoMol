
## Train
- `train.py`: used for training, core script
```bash
python 
```
- `model`: contains many scripts supporting training and testing
- `manager_torch.py`: script for selecting appropriate GPU card in a machine with >1 GPU cards
## Test

### Utils
- `pkl2sdf.py`: mol from .pickle to .sdf, for visualization
`python pkl2sdf.py 

- `sdf2pkl.py`: mol from .sdf to .pickle, for analysis
`python

### Conformer-Generation
- `gen_GeoMol.py`: for smi, rdkit mol and GeoMol mol generation from pickle file (in situ validation)
```bash
python gen_GeoMol.py \
--split split0 \
--dataset drugs \
--n_true_confs 20 \
--n_testset 1000 \
--rdkit \
--geomol \
--trained_model_dir /pubhome/qcxia02/git-repo/AI-CONF/GeoMol/trained_models/drugs \
--datatype test
```
### Metric
- `get_results.py`: for result evaluation and output
```bash
testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-split0/12-11-15-37
python get_results.py \
--testpath $testpath \
--core 2 \
--gen test_GeoMol_20.pickle
# --rdkit test_rdkit_20.pickle

```

- `gen_rdkit.py`: generate rdkit mols solely from `test_ref.pickle`


- `generate_confs.py`: generate GeoMol mols solely from `test_smiles.csv`(batch) or `smi, nconfs`(single)
```bash
# Example 1. batch
testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-split0/12-20-11-25
trained_model_dir=/tmp/drugs-split0-e100-b16-d50
python generate_confs.py \
--trained_model_dir $trained_model_dir \
--test_csv $testpath/test_smiles.csv \
--dataset drugs \
--out $testpath/test_GeoMol.pickle
# Example 2. single
python generate_confs.py \
--trained_model_dir /pubhome/qcxia02/git-repo/AI-CONF/GeoMol/trained_models/drugs \
--smi C1CCCCC1 \
--numgenconfs 5 \
--dataset drugs \
--out test_GeoMol.pickle
***
- All shell scripts are used for SGE submit of these python scripts