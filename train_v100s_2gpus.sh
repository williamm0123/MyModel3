#!/bin/bash -l
#SBATCH --job-name=MyModel3_pipeline
#SBATCH --partition=gpu-v100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# 1) 确保日志目录存在（否则可能直接退出）
mkdir -p logs

echo "JobID=$SLURM_JOB_ID on $(hostname -s)"
date
nvidia-smi || true

# 2) 你的运行指令
export DATA_ROOT=/scr/user/qinglong
export CUDA_VISIBLE_DEVICES=0,1

python -u train.py --config config/mvs.json