# -*- coding:utf-8 -*-
# author: Awet
# @file: concordance_wod.py


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
    source_dir = args.source_dir
    teacher_1 = args.teacher1
    teacher_2 = args.teacher2
    teacher_3 = args.teacher3
    lamda  = args.lamda

    #sequence = ["00", "01", "02","03", "04", "05", "06", "07", "09", "10"]

    sequence = os.listdir(source_dir)

    for i, sq in enumerate(sequence):
        preds_t1 = sorted(glob.glob(os.path.join(source_dir, sq, f"predictions_{teacher_1}", '*.npy')))
        preds_t2 = sorted(glob.glob(os.path.join(source_dir, sq, f"predictions_{teacher_2}", '*.npy')))

        probs_t1 = sorted(glob.glob(os.path.join(source_dir, sq, f"probability_{teacher_1}", '*.npy')))
        probs_t2 = sorted(glob.glob(os.path.join(source_dir, sq, f"probability_{teacher_2}", '*.npy')))
        if teacher_3 is not None:
            preds_t3 = sorted(glob.glob(os.path.join(source_dir, sq, f"predictions_{teacher_3}", '*.npy')))
            probs_t3 = sorted(glob.glob(os.path.join(source_dir, sq, f"probability_{teacher_3}", '*.npy')))

        frame_len = len(preds_t1)

        for frame in range(frame_len):
            frame_name = str(frame).zfill(6)

            pred = np.array([np.load(preds_t1[frame]).reshape((-1, 1)), np.load(preds_t2[frame]).reshape((-1, 1))])
            prob = np.array([np.load(probs_t1[frame]).reshape((-1, 1)), np.load(probs_t2[frame]).reshape((-1, 1))])
            if teacher_3 is not None:
                pred = np.array([np.load(preds_t1[frame]).reshape((-1, 1)), np.load(preds_t2[frame]).reshape((-1, 1)), np.load(preds_t3[frame]).reshape((-1, 1))])
                prob = np.array([np.load(probs_t1[frame]).reshape((-1, 1)), np.load(probs_t2[frame]).reshape((-1, 1)), np.load(probs_t3[frame]).reshape((-1, 1))])


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
            '''
            if not os.path.exists(os.path.join(source_dir, sq, f"predictions_f11_33")):
                os.makedirs(os.path.join(source_dir, sq, f"predictions_f11_33"))
                os.makedirs(os.path.join(source_dir, sq, f"probability_f11_33"))
            '''

            # best_pred.tofile(os.path.join(source_dir, sq, f"predictions_f11_33", frame_name + '.npy'))
            # new_prob.tofile(os.path.join(source_dir, sq,  f"probability_f11_33", frame_name + '.npy'))
            np.save(os.path.join(source_dir, sq, f"predictions_f11_33", frame_name), np.squeeze(best_pred))
            np.save(os.path.join(source_dir, sq,  f"probability_f11_33", frame_name), np.squeeze(new_prob))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lamda', default=0.1, type=float)
    parser.add_argument('-b', '--best', default=True)
    parser.add_argument('-s', '--source_dir', default='/mnt/beegfs/gpu/argoverse-tracking-all-training/WOD/processed/Unlabeled/training')
    parser.add_argument('-x', '--teacher1', required=False, default="f1_1")
    parser.add_argument('-y', '--teacher2', required=False, default="f2_2")
    parser.add_argument('-z', '--teacher3', default="f3_3")
    args = parser.parse_args()

    main(args)
