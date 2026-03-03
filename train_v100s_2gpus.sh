#!/bin/bash -l
#SBATCH --job-name=mvs_01
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --qos=long
#SBATCH --time=3-0:0:0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

source ~/.bashrc
conda activate mvs

DATA_ROOT=/scr/user/qinglong CUDA_VISIBLE_DEVICES=0,1 \
python -u train.py --config config/mvsformer++.json \
  --exp_name MVSFormerpp01 \
  --DDP