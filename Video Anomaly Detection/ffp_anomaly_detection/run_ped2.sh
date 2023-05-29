#!/bin/bash
#SBATCH -J ahnsunghyun_vad
#SBATCH -o run_ped2.txt
echo "### START DATE=$(date)"
python train.py --dataset=ped2 --flownet=2sd --save_dir=sha --work_num=1
python evaluate.py --dataset=ped2 --trained_model=latest_ped2_80001.pth
python evaluate.py --dataset=ped2 --trained_model=1_best_ped2.pth
echo "### END DATE=$(date)"