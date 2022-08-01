# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# at 5/17/22
# --------------------------|
import numpy as np

import os
import glob
dest_dir = '/mnt/beegfs/gpu/argoverse-tracking-all-training/WOD/processed/Labeled/cut_mix'
source_dir = '/mnt/beegfs/gpu/argoverse-tracking-all-training/WOD/processed/Labeled/training'
sequence = os.listdir(source_dir)
len(sequence)

motor_cyclist_pcl = []
motor_cyclist_ss = []
id = 0
for seq in sequence:
    frames = glob.glob(f"{source_dir}/{seq}/lidar/*.npy")
    for f in frames:
        pcl = np.load(f)
        ss = np.load(f.replace('lidar', 'labels_v3_2'))
        motor_cyclist_mask = ss[:,1] == 4

        if np.sum(motor_cyclist_mask) > 10:
            # extract the pcl and label
            pcl_m = pcl[motor_cyclist_mask]
            ss_m = ss[motor_cyclist_mask]
            ss_m[:,0]= id
            motor_cyclist_pcl.append(pcl_m)
            motor_cyclist_ss.append(ss_m)
            #np.save(os.path.join(dest_dir, 'lidar', str(id).zfill(6)), pcl_m)
            #np.save(os.path.join(dest_dir, 'labels_v3_2', str(id).zfill(6)), ss_m)
            id += 1

motor_cyclist_pcl_all = np.concatenate(motor_cyclist_pcl, axis=0)
motor_cyclist_ss_all = np.concatenate(motor_cyclist_ss, axis=0)

np.save(os.path.join(dest_dir, 'pcl_o_vehicles'), motor_cyclist_pcl_all)
np.save(os.path.join(dest_dir, 'ss_id_o_vehicles'), motor_cyclist_ss_all)

print("Process Finished Successfully")





