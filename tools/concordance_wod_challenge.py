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
    destination_dir = args.destination_dir
    teacher_1 = args.teacher1
    teacher_2 = args.teacher2
    teacher_3 = args.teacher3
    lamda  = args.lamda
    teachers = 0
    if teacher_1 is not None:    
        teachers +=1
    if teacher_2 is not None:    
        teachers +=1    
    if teacher_3 is not None:    
        teachers +=1
    #sequence = ["00", "01", "02","03", "04", "05", "06", "07", "09", "10"]
    print("teachers: {teachers}")

    sequence = os.listdir(teacher_1)

    for i, sq in enumerate(sequence):
        preds_t1 = sorted(glob.glob(os.path.join(teacher_1, sq, f"prediction", '*.npy')))
        preds_t2 = sorted(glob.glob(os.path.join(teacher_2, sq, f"prediction", '*.npy')))

        probs_t1 = sorted(glob.glob(os.path.join(teacher_1, sq, f"probability", '*.npy')))
        probs_t2 = sorted(glob.glob(os.path.join(teacher_2, sq, f"probability", '*.npy')))
        if teacher_3 is not None:
            preds_t3 = sorted(glob.glob(os.path.join(teacher_3, sq, f"prediction", '*.npy')))
            probs_t3 = sorted(glob.glob(os.path.join(teacher_3, sq, f"probability", '*.npy')))
   

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

            for i in range(teachers):
                preded = pred[i]
                concored = best_pred == preded
                weight += concored.astype(int)

            best_prob = max_pob
            new_prob = best_prob + ((weight-1) * lamda)
            new_prob = np.minimum(np.ones_like(best_prob), new_prob)
            new_prob = new_prob.astype(np.float32)

            if not os.path.exists(os.path.join(destination_dir, sq, f"prediction")):
                os.makedirs(os.path.join(destination_dir, sq, f"prediction"))
                os.makedirs(os.path.join(destination_dir, sq, f"probability"))

            # best_pred.tofile(os.path.join(destination_dir, sq, f"predictions_f11_33", frame_name + '.npy'))
            # new_prob.tofile(os.path.join(destination_dir, sq,  f"probability_f11_33", frame_name + '.npy'))
            np.save(os.path.join(destination_dir, sq, f"prediction", frame_name), np.squeeze(best_pred))
            np.save(os.path.join(destination_dir, sq,  f"probability", frame_name), np.squeeze(new_prob))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lamda', default=0.1, type=float)
    parser.add_argument('-b', '--best', default=True)
    parser.add_argument('-s', '--destination_dir', default='/mnt/beegfs/gpu/argoverse-tracking-all-training/WOD/challenge_v3_2/test/f10_30/test')
    parser.add_argument('-x', '--teacher1', required=False, default="/mnt/beegfs/gpu/argoverse-tracking-all-training/WOD/challenge_v3_2/test/f1_0/test")
    parser.add_argument('-y', '--teacher2', required=False, default="/mnt/beegfs/gpu/argoverse-tracking-all-training/WOD/challenge_v3_2/test/f2_0/test")
    #parser.add_argument('-z', '--teacher3', default=None)    
    parser.add_argument('-z', '--teacher3', default="/mnt/beegfs/gpu/argoverse-tracking-all-training/WOD/challenge_v3_2/test/f3_0/test")
    args = parser.parse_args()

    main(args)
