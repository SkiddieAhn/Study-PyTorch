#!/bin/bash
#SBATCH -J ahnsunghyun_vad
#SBATCH -o sbatch_result/run_ped2.txt
echo "### START DATE=$(date)"
python train.py --dataset=ped2 --generator=pm1 --flownet=2sd --save_dir=sha --work_num=2
python evaluate.py --dataset=ped2 --generator=pm1 --trained_model=latest_ped2_100001.pth
python evaluate.py --dataset=ped2 --generator=pm1 --trained_model=2_best_ped2.pth
echo "### END DATE=$(date)"