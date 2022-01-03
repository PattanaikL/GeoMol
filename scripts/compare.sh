##$ -S /bin/bash
###$ -q mazda
##$ -pe mazda 32
####$ -l ngpus=1
##$ -N conformer_gen_CPU
##$ -M xiaqiancheng@nibs.ac.cn
##$ -o /pubhome/qcxia02/git-repo/AI-CONF/GeoMol/out
##$ -e /pubhome/qcxia02/git-repo/AI-CONF/GeoMol/error
##$ -cwd
### $ -now y

conda activate GeoMol-cuda11x

'''
Usage:
# python get_results.py --testpath $testpath --core 1 --gen test_GeoMol_20.pickle --rdkit test_rdkit_20.pickle --no 1

# testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-split0/12-11-15-37
# testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-split0/12-19-20-5
# testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-plati/train_5epoch
# testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-split0/12-20-11-25
# testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-split0/12-20-13-00
# testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-split0/12-20-13-00
# testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-plati/pre-train

core=16
# mols1=test_GeoMol_50.pickle
mols1=test_GeoMol_50_cutoff0.7.pickle
mols2=test_rdkit_50.pickle
rdkerrtxt=rdkit_err_smiles_25.txt
# mols1=test_GeoMol.pickle
# mols2=test_rdkit.pickle
# rdkerrtxt=rdkit_err_smiles.txt
geoerrtxt=GeoMol_err_smiles.txt
maxmatches=100
queue=k230

# testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-split0/12-20-11-25
testpath=/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-plati/pre-train

# for threshold in `seq 2 5`; do
for threshold in 5; do
qsub_anywhere.py -c "source compare.sh $testpath $core $mols1 $mols2 $threshold $rdkerrtxt $geoerrtxt" -q $queue -n $core -j . -N c$core-$mols1-$mols2-t$threshold-m100 --qsub_now
done
'''

testpath=$1
core=$2
mols1=$3
mols2=$4
threshold=$5
rdkerrtxt=$6
geoerrtxt=$7

python compare.py --testpath $testpath --mols1 $mols1 --mols2 $mols2 --threshold $threshold --core $core --rdkerrtxt $rdkerrtxt --geoerrtxt $geoerrtxt --maxmatches 100
python compare.py --testpath $testpath --mols1 $mols1 --mols2 $mols2 --threshold $threshold --core $core --rdkerrtxt $rdkerrtxt --geoerrtxt $geoerrtxt --maxmatches 100 --removeH


