#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=24                  # 1 node
#SBATCH --ntasks-per-node=1         # 36 tasks per node
#SBATCH --time=10-00:00:0              # time limits: 500 hour
#SBATCH --partition=amdgpuextralong	  # gpufast
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --output=/home/gebreawe/Model_logs/Segmentation/T-UDA/logs/train_uda_nuscenes_wod_f2_0_time_%j.log     # file name for stdout/stderr
# module
#ml spconv/20210618-fosscuda-2020b
ml spconv/2.1.21-foss-2021a-CUDA-11.3.1
ml PyTorch-Geometric/2.0.2-foss-2021a-CUDA-11.3.1-PyTorch-1.10.0

cd ../..

name=t-uda

#CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 python -u train_cylinder_asym.py --mgpus \
#2>&1 | tee logs_dir/${name}_logs_tee_m_b4.txt

export NCCL_LL_THRESHOLD=0

echo "epoch 0"

#CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM train_uda.py --config_path 'configs/data_config/da_wod_nuscenes/uda_wod_nuscenes_f2_0_time.yaml' 2>&1 | tee logs_dir/${name}_logs_uda_wod_nuscenes_f2_0_time.txt

python train_uda.py --config_path 'configs/data_config/da_wod_nuscenes/uda_nuscenes_wod_f2_0_time.yaml' 2>&1 | tee logs_dir/${name}_uda_nuscenes_wod_f2_0_time.txt
