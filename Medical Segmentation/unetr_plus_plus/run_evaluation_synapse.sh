#!/bin/sh

DATASET_PATH=DATASET
CHECKPOINT_PATH=unetr_pp/evaluation/unetr_pp_synapse_checkpoint

export PYTHONPATH=/usr/bin/python3
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python3 unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_synapse 2 0 -val
