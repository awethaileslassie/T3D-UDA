# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys

import numpy as np
import glob
import os
import shutil
import random
import math


def main(args):
    #sequence = ["04"]
    #des_seq = ["34"]
    teacher_1 = args.teacher1
    teacher_2 = args.teacher2
    teacher_3 = args.teacher3
    lamda  = args.lamda
    concordance = args.concordance

    sequence = ["00", "01", "02","03", "04", "05", "06", "07", "09", "10"]
    #des_seq = ["11", "12", "13","14", "15", "16", "17", "18", "19", "20"]

    source = '/mnt/beegfs/gpu/argoverse-tracking-all-training/semantic-kitti/train_pseudo_20/sequences'
    destination = '/mnt/beegfs/gpu/argoverse-tracking-all-training/semantic-kitti/train_pseudo_20/sequences'

    for i, sq in enumerate(sequence):
        preds_t1 = sorted(glob.glob(os.path.join(source, sq, f"predictions_{teacher_1}", '*.label')))
        preds_t2 = sorted(glob.glob(os.path.join(source, sq, f"predictions_{teacher_2}", '*.label')))

        probs_t1 = sorted(glob.glob(os.path.join(source, sq, f"probability_{teacher_1}", '*.label')))
        probs_t2 = sorted(glob.glob(os.path.join(source, sq, f"probability_{teacher_2}", '*.label')))
        if teacher_3 is not None:
            preds_t3 = sorted(glob.glob(os.path.join(source, sq, f"predictions_{teacher_3}", '*.label')))
            probs_t3 = sorted(glob.glob(os.path.join(source, sq, f"probability_{teacher_3}", '*.label')))

        frame_len = len(preds_t1)

        for frame in range(frame_len):
            frame_name = str(frame).zfill(6)

            pred = np.array([np.fromfile(preds_t1[frame], dtype=np.int32).reshape((-1, 1)), np.fromfile(preds_t2[frame], dtype=np.int32).reshape((-1, 1))])
            prob = np.array([np.fromfile(probs_t1[frame], dtype=np.float32).reshape((-1, 1)), np.fromfile(probs_t2[frame], dtype=np.float32).reshape((-1, 1))])
            if teacher_3 is not None:
                pred = np.array([np.fromfile(preds_t1[frame], dtype=np.int32).reshape((-1, 1)), np.fromfile(preds_t2[frame], dtype=np.int32).reshape((-1, 1)), np.fromfile(preds_t3[frame], dtype=np.int32).reshape((-1, 1))])
                prob = np.array([np.fromfile(probs_t1[frame], dtype=np.float32).reshape((-1, 1)), np.fromfile(probs_t2[frame], dtype=np.float32).reshape((-1, 1)), np.fromfile(probs_t3[frame], dtype=np.float32).reshape((-1, 1))])


            max_pob = prob.max(axis=0)
            max_pob_id = prob.argmax(axis=0)
            best_pred = np.zeros_like(pred[0])
            for i in range(len(max_pob)):
                best_pred[i] = pred[int(max_pob_id[i]), i]


            weight = np.zeros_like(best_pred)

            for i in range(3):
                preded = pred[i]
                concored = best_pred == preded
                weight += concored.astype(int)

            best_prob = max_pob
            new_prob = best_prob + ((weight-1) * lamda)
            new_prob = np.minimum(np.ones_like(best_prob), new_prob)
            new_prob = new_prob.astype(np.float32)

            if not os.path.exists(os.path.join(destination, sq, f"predictions_{concordance}")):
                os.makedirs(os.path.join(destination, sq, f"predictions_{concordance}"))
                os.makedirs(os.path.join(destination, sq, f"probability_{concordance}"))

            best_pred.tofile(os.path.join(destination, sq, f"predictions_{concordance}", frame_name + '.label'))
            new_prob.tofile(os.path.join(destination, sq,  f"probability_{concordance}", frame_name + '.label'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lamda', default=0.1, type=float)
    parser.add_argument('-b', '--best', default=True, )
    parser.add_argument('-x', '--teacher1', required=False, default="f3_3")
    parser.add_argument('-y', '--teacher2', required=False, default="f3_32")
    parser.add_argument('-z', '--teacher3', default="f3_33")
    parser.add_argument('-c', '--concordance', default="f333_333")
    args = parser.parse_args()

    main(args)
