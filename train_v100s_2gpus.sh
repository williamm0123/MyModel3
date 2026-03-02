#!/bin/bash -l
#SBATCH --job-name=MyModel3_v100s_2gpu
#SBATCH --partition=gpu-v100s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs
export DATA_ROOT=/scr/user/qinglong
export CUDA_VISIBLE_DEVICES=0,1
cd /home/user/qinglong/MyModel3
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 /home/user/qinglong/MyModel3/train.py --config config/mvs.json