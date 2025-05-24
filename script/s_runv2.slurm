#!/usr/bin/env bash
#SBATCH -J prompt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -o csl-out.%j        
#SBATCH -e csl-err.%j
#SBATCH --partition=A40_32C
#!/usr/bin/env bash

clear
python main.py
# CUDA_VISIBLE_DEVICES=0 nohup python main.py > nk_deppro.txt &
# CUDA_VISIBLE_DEVICES=0 python main.py
# csl-err.%j
# --nodelist=gpu4