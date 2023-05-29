#!/bin/bash
#SBATCH -J ahnsunghyun_original_test
#SBATCH -o original_test.txt
echo "### START DATE=$(date)"
echo "### HOSTNAME=\%(hostname)"
python -u ltr/run_training.py transt transt
echo "### END DATE=$(date)"