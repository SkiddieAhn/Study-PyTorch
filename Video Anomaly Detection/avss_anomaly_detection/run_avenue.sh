#!/bin/bash
#SBATCH -J ahnsunghyun_vad
#SBATCH -o sbatch_result/run_avenue.txt
echo "### START DATE=$(date)"
python train.py --dataset=avenue --generator=pm1 --flownet=2sd --save_dir=sha --work_num=1
python evaluate.py --dataset=avenue --generator=pm1 --trained_model=latest_avenue_100001.pth
python evaluate.py --dataset=avenue --generator=pm1 --trained_model=1_best_avenue.pth
echo "### END DATE=$(date)"