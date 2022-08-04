# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiowt


import glob
import os


def main():
    source = '/mnt/beegfs/gpu/argoverse-tracking-all-training/Synth4D/nuscenes_synth'

    sequence = os.listdir(source)

    for i, sq in enumerate(sequence):
        files = sorted(glob.glob(os.path.join(source, sq, "velodyne", '*.npy')))
        total_frame = len(files)

        for frame in range(total_frame):
            frame_name = str(frame).zfill(6)
            dest_lidar = os.path.join(source, sq, "lidar")
            if not os.path.exists(dest_lidar):
                os.mkdir(dest_lidar)
            dest_labels = os.path.join(source, sq, "labels")
            if not os.path.exists(dest_labels):
                os.mkdir(dest_labels)
            dest_poses = os.path.join(source, sq, "poses")
            if not os.path.exists(dest_poses):
                os.mkdir(dest_poses)

            os.rename(os.path.join(source, sq, "velodyne", str(frame) + '.npy'),
                      os.path.join(dest_lidar, frame_name + '.npy'))
            os.rename(os.path.join(source, sq, "labels", str(frame) + '.npy'),
                      os.path.join(dest_labels, frame_name + '.npy'))
            os.rename(os.path.join(source, sq, "calib", str(frame) + '.npy'),
                      os.path.join(dest_poses, frame_name + '.npy'))


if __name__ == '__main__':
    main()
    print(f"------------------------------Task finished-------------------------")
