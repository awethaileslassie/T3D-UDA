#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=24                  # 1 node
#SBATCH --ntasks-per-node=1         # 36 tasks per node
#SBATCH --time=21-00:00:0              # intensity limits: 500 hour
#SBATCH --partition=amdgpuextralong	  # gpufast
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --output=/home/gebreawe/Model_logs/Segmentation/ST-UDA/logs/wod_train_f0_0_all_v3_2_intensity_beam32_%j.log     # file name for stdout/stderr
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

echo "epoch 0"

#CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 python -u -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM train_cylinder_asym_wod.py --config_path 'configs/wod/wod_f0_0.yaml' 2>&1 | tee logs_dir/${name}_logs_wod_f0_0_b2_v3_2.txt

python train_wod.py --config_path 'configs/wod/wod_f0_0_intensity_beam32.yaml' 2>&1 | tee logs_dir/${name}_logs_wod_f0_0_b2_v3_2_intensity_beam32.txt
