#!/bin/bash
'''
Usage:
trained_model_dir=$HOME/git-repo/AI-CONF/GeoMol/trained_models/drugs
datatype=val
qsub_anywhere.py -c "source gen_GeoMol.sh $trained_model_dir $datatype" -q k233 -n 16 -j . -N gen_conf_$datatype --qsub_now
'''
conda activate GeoMol-cuda11x

trained_model_dir=$1
datatype=$2

python gen_GeoMol.py \
--split split0 \
--dataset drugs \
--n_true_confs 20 \
--n_testset 1000 \
--rdkit \
--geomol \
--trained_model_dir $trained_model_dir \
--datatype $datatype