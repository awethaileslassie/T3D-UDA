# -*- coding:utf-8 -*-
# author: Awet


import glob
import math
import os
import shutil

import numpy as np


def main():
    # sequence = ["04"]
    # des_seq = ["15"]
    dat_path = '/mnt/personal/gebreawe/Datasets/Synthetic/Valeo/Livox/processed'
    split_percent = 10  # 10 #20
    source = f"{dat_path}/train"
    destination = f"{dat_path}/val"

    count = 0
    percent = split_percent / 100.0

    files = glob.glob(os.path.join(source, "lidar", '*.npy'))
    toatl_frame = len(files)
    number_frame = math.ceil(toatl_frame * percent)

    sampled_frame = range(toatl_frame)[-number_frame:]
    print(sampled_frame)
    if not os.path.exists(os.path.join(destination, "lidar")):
        os.makedirs(os.path.join(destination,  "lidar"))
        os.makedirs(os.path.join(destination, "labels"))

    for f, frame in enumerate(sampled_frame):
        frame_name = str(frame).zfill(6)
        f_name = str(f).zfill(6)

        shutil.move(os.path.join(source, "lidar", frame_name + '.npy'),
                    os.path.join(destination, "lidar", f_name + '.npy'))
        shutil.move(os.path.join(source, "labels", frame_name + '.npy'),
                    os.path.join(destination, "labels", f_name + '.npy'))


if __name__ == '__main__':
    main()
    print(f"------------------------------Task finished-------------------------")
