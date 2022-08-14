# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# at 8/10/22
# --------------------------|
import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from builder import data_builder, model_builder, loss_builder
from configs.config import load_config_data
from dataloader.pc_dataset import get_label_name, update_config
from utils.load_save_util import load_checkpoint
from utils.metric_util import per_class_iu, fast_hist_crop
from utils.per_class_weight import semantic_kitti_class_weights
import copy

def yield_target_dataset_loader(n_epochs, target_train_dataset_loader):
    for e in range(n_epochs):
        for i_iter_train, (_, train_vox_label, train_grid, _, train_pt_fea, ref_st_idx, ref_end_idx, lcw) \
                in enumerate(target_train_dataset_loader):
            yield train_vox_label, train_grid, train_pt_fea, ref_st_idx, ref_end_idx, lcw

class Trainer(object):
    def __init__(self, student_model, teacher_model, optimizer, ckpt_dir, unique_label, unique_label_str,
                 lovasz_softmax,
                 loss_func,
                 ignore_label, train_mode=None, ssl=None, eval_frequency=1, pytorch_device=0, warmup_epoch=1,
                 ema_frequency=5):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.model_save_path = ckpt_dir
        self.unique_label = unique_label
        self.unique_label_str = unique_label_str
        self.eval_frequency = eval_frequency
        self.lovasz_softmax = lovasz_softmax
        self.loss_func = loss_func
        self.ignore_label = ignore_label
        self.train_mode = train_mode
        self.ssl = ssl
        self.pytorch_device = pytorch_device
        self.warmup_epoch = warmup_epoch
        self.ema_frequency = ema_frequency

