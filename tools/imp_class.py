# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from builder import model_builder
from config.config import load_config_data
from dataloader.pc_dataset import get_SemKITTI_label_name, update_config
from utils.load_save_util import load_checkpoint

warnings.filterwarnings("ignore")

# training
epoch = 0
best_val_miou = 0
global_iter = 0

number_sample = {0: 43912468, 1: 689307, 10: 95302518, 11: 391766, 13: 65505, 15: 936031, 16: 0, 18: 4347360,
                 20: 5020865, 30: 440239, 31: 5, 32: 13, 40: 466974969, 44: 34577789, 48: 338183720, 49: 9173976,
                 50: 311802516, 51: 170001681, 52: 5613664, 60: 110620, 70: 627195745, 71: 14189414, 72: 183603141,
                 80: 6712285, 81: 1441988, 99: 23371792, 252: 4128968, 253: 298599, 254: 376574, 255: 87766, 256: 0,
                 257: 266513, 258: 238730, 259: 103005}
class_name = {0: 0,  # "unlabeled", and others ignored
              1: 10,  # "car"
              2: 11,  # "bicycle"
              3: 15,  # "motorcycle"
              4: 18,  # "truck"
              5: 20,  # "other-vehicle"
              6: 30,  # "person"
              7: 31,  # "bicyclist"
              8: 32,  # "motorcyclist"
              9: 40,  # "road"
              10: 44,  # "parking"
              11: 48,  # "sidewalk"
              12: 49,  # "other-ground"
              13: 50,  # "building"
              14: 51,  # "fence"
              15: 70,  # "vegetation"
              16: 71,  # "trunk"
              17: 72,  # "terrain"
              18: 80,  # "pole"
              19: 81  # "traffic-sign"
              }

learning_map = {
    0: 0,  # "unlabeled"
    1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,  # "car"
    11: 2,  # "bicycle"
    13: 5,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,  # "motorcycle"
    16: 5,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,  # "truck"
    20: 5,  # "other-vehicle"
    30: 6,  # "person"
    31: 7,  # "bicyclist"
    32: 8,  # "motorcyclist"
    40: 9,  # "road"
    44: 10,  # "parking"
    48: 11,  # "sidewalk"
    49: 12,  # "other-ground"
    50: 13,  # "building"
    51: 14,  # "fence"
    52: 0,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,  # "lane-marking" to "road" ---------------------------------mapped
    70: 15,  # "vegetation"
    71: 16,  # "trunk"
    72: 17,  # "terrain"
    80: 18,  # "pole"
    81: 19,  # "traffic-sign"
    99: 0,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,  # "moving-car" to "car" ------------------------------------mapped
    253: 7,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,  # "moving-person" to "person" ------------------------------mapped
    255: 8,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,  # "moving-truck" to "truck" --------------------------------mapped
    259: 5,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}
class_weight = []


def main(args):
    # pytorch_device = torch.device("cuda:2") # torch.device('cuda:2')
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'true'
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '9994'
    # os.environ['RANK'] = "0"
    os.environ['OMP_NUM_THREADS'] = "1"

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    print(f"distributed: {distributed}")

    pytorch_device = args.local_rank
    if distributed:
        torch.cuda.set_device(pytorch_device)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    config_path = args.config_path

    configs = load_config_data(config_path)

    # send config parameters to pc_dataset
    update_config(configs)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    past_frame = train_hypers['past']
    future_frame = train_hypers['future']
    ssl = train_hypers['ssl']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config).to(pytorch_device)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    # if args.mgpus:
    #     my_model = nn.DataParallel(my_model)
    #     #my_model.cuda()
    # #my_model.cuda()

    # my_model = my_model().to(pytorch_device)
    # if args.local_rank >= 1:
    if distributed:
        my_model = DistributedDataParallel(
            my_model,
            device_ids=[pytorch_device],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    # for weighted class loss
    weighted_class = False

    # for focal loss
    focal_loss = True

    # 20 class number of samples from training sample

    # newly computed from learning_inv_map
    per_class_count = np.array(
        [7.35872310e+07, 9.94314860e+07, 3.91766000e+05, 9.36031000e+05,
         4.58609000e+06, 5.45588800e+06, 8.16813000e+05, 2.98604000e+05,
         8.77790000e+04, 4.67085589e+08, 3.45777890e+07, 3.38183720e+08,
         9.17397600e+06, 3.11802516e+08, 1.70001681e+08, 6.27195745e+08,
         1.41894140e+07, 1.83603141e+08, 6.71228500e+06, 1.44198800e+06])

    f1_0 = np.array([95.22, 20.10, 54.25, 43.92, 32.22, 57.62, 73.94, 0.01,
                      93.27, 37.09,
                      78.23, 0.92,
                      87.46, 42.91,
                      85.14, 63.11,
                      68.35, 63.53,
                      49.77, 55.11, ])

    f1_T11_33 = np.array([94.91, 33.67,
                          53.10, 48.65,
                          49.99,
                          74.62,
                          85.86,
                          0.00,
                          93.40,
                          42.72,
                          79.92,
                          1.35,
                          90.90,
                          59.57,
                          89.07,
                          63.29,
                          77.01,
                          62.33,
                          50.11,
                          60.55])
                          
    f2_0 = np.array([93.17,30.99,47.51,24.47,31.54,62.33,78.9,0.00,93.01,
    39.34,78.21,1.1,88.01,47.2,86.78,59.08,72.66,63.93,47.72,55.051])
    
    f2_T11_33 = np.array([94.6,45.21,59.01,45.42,45.29,74.31,83.46,0.00,
    92.34,39.53,77.43,0.29,90.85,57.99,87.93,66.47,72.63,64.08,49.76,60.348])

    imp1 = f1_T11_33 - f1_0
    imp2 = f2_T11_33 - f2_0
    fig = plt.Figure()
    plt.scatter(range(19), per_class_count[1:], marker='+', label="improvemnt")
    fig = plt.Figure()
    plt.scatter(range(19), per_class_count[1:], marker='*', label="improvemnt")
    plt.show()

    # per class/object number
    print(f"number of class: {per_class_count}")


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti_f1_0.yaml')
    parser.add_argument('-g', '--mgpus', action='store_true', default=False)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
