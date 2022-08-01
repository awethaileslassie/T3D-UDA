#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16                  # 1 node
#SBATCH --ntasks-per-node=1       # 36 tasks per node
#SBATCH --time=4:00:00               # time limits: 500 hour
#SBATCH --partition=amdgpufast	  # gpufast
#SBATCH --gres=gpu:2
#SBATCH --mem=40G
#SBATCH --output=run_val_f3_3_wod_kitti_%j.log     # file name for stdout/stderr
# module
#ml spconv/20210618-fosscuda-2020b
ml spconv/2.1.21-foss-2021a-CUDA-11.3.1
ml PyTorch-Geometric/2.0.2-foss-2021a-CUDA-11.3.1-PyTorch-1.10.0

cd ../..

name=cylinder_asym_networks

#gpuid=0,1
#CUDA_VISIBLE_DEVICES=${gpuid}  python -u train_cylinder_asym.py \
#2>&1 | tee logs_dir/${name}_logs_tee_b12.txt

#CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 python -u train_cylinder_asym.py --mgpus \
#2>&1 | tee logs_dir/${name}_logs_tee_m_b4.txt

export NCCL_LL_THRESHOLD=0

python test_cylinder_asym.py --config_path 'config/da_wod_kitti/da_wod_kitti_T3_3.yaml' --mode 'val' --save 'True' 2>&1 | tee logs_dir/${name}_logs_val_T3_3_wod_kitti.txt


