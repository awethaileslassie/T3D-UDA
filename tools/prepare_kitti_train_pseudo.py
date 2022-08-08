# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


import glob
import math
import os
import shutil

import numpy as np


def main():
    # sequence = ["04"]
    # des_seq = ["15"]
    sequence = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
    des_seq = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
    # sequence = ["04", "05", "06", "07", "09", "10"]
    # des_seq =  ["15", "16", "17", "18", "19", "20"]

    split_percent = 40  # 10 #20
    source = f'/mnt/beegfs/gpu/argoverse-tracking-all-training/semantic-kitti/train_pseudo_{int(split_percent)}/sequences'
    destination = f'/mnt/beegfs/gpu/argoverse-tracking-all-training/semantic-kitti/train_pseudo_{int(split_percent)}/sequences'

    count = 0

    percent = split_percent / 100.0

    for i, sq in enumerate(sequence):
        files = glob.glob(os.path.join(source, sq, "velodyne", '*.bin'))
        toatl_frame = len(files)
        number_frame = math.ceil(toatl_frame * percent)

        sampled_frame = range(number_frame)

        if not os.path.exists(os.path.join(destination, sq, "velodyne")):
            os.makedirs(os.path.join(destination, sq, "velodyne"))
            os.makedirs(os.path.join(destination, sq, "labels"))

        if not os.path.exists(os.path.join(destination, des_seq[i], "velodyne")):
            os.makedirs(os.path.join(destination, des_seq[i], "velodyne"))
            os.makedirs(os.path.join(destination, des_seq[i], "labels"))

        for frame in sampled_frame:
            frame_name = str(frame).zfill(6)

            shutil.move(os.path.join(source, sq, "velodyne", frame_name + '.bin'),
                        os.path.join(destination, des_seq[i], "velodyne", frame_name + '.bin'))
            shutil.move(os.path.join(source, sq, "labels", frame_name + '.label'),
                        os.path.join(destination, des_seq[i], "labels", frame_name + '.label'))

        shutil.copy(os.path.join(source, sq, "calib.txt"), os.path.join(destination, des_seq[i], "calib.txt"))

        poses = np.loadtxt(os.path.join(source, sq, "poses.txt"))
        dest_pose = poses[:number_frame]
        source_pose = poses[number_frame:]
        times = np.loadtxt(os.path.join(source, sq, "times.txt"))
        dest_time = times[:number_frame]
        source_time = times[number_frame:]
        # shutil.copy(os.path.join(source, sq, "poses.txt"), os.path.join(destination, des_seq[i], "poses.txt"))
        # shutil.copy(os.path.join(source, sq, "times.txt"), os.path.join( destination, des_seq[i], "times.txt"))
        np.savetxt(os.path.join(destination, des_seq[i], "poses.txt"), dest_pose)
        np.savetxt(os.path.join(source, sq, "poses.txt"), source_pose)

        np.savetxt(os.path.join(destination, des_seq[i], "times.txt"), dest_time)
        np.savetxt(os.path.join(source, sq, "times.txt"), source_time)


if __name__ == '__main__':
    main()
    print(f"------------------------------Task finished-------------------------")
