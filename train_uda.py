# -*- coding:utf-8 -*-
# author: Awet

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
from utils.trainer_function import Trainer
import copy

warnings.filterwarnings("ignore")

# clear/empty cached memory used by caching allocator
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

# training
epoch = 0
best_val_miou = 0
global_iter = 0


def main(args):
    # pytorch_device = torch.device("cuda:2") # torch.device('cuda:2')
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'true'
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '9994'
    # os.environ['RANK'] = "0"
    # If your script expects `--local_rank` argument to be set, please
    # change it to read from `os.environ['LOCAL_RANK']` instead.
    # args.local_rank = os.environ['LOCAL_RANK']

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

    # send configs parameters to pc_dataset
    update_config(configs)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']
    ssl_dataloader_config = configs['ssl_data_loader']

    source_val_batch_size = val_dataloader_config['batch_size']
    source_train_batch_size = train_dataloader_config['batch_size']
    target_train_batch_size = ssl_dataloader_config['batch_size']

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

    SemKITTI_label_name = get_label_name(dataset_config["label_mapping"])
    # NB: no ignored class
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    student_model = model_builder.build(model_config).to(pytorch_device)

    teacher_model = model_builder.build(model_config).to(pytorch_device)

    if os.path.exists(model_load_path):
        student_model = load_checkpoint(model_load_path, student_model, map_location=pytorch_device)
        teacher_model = load_checkpoint(model_load_path, teacher_model, map_location=pytorch_device)

    # if args.mgpus:
    #     student_model = nn.DataParallel(student_model)
    #     #student_model.cuda()
    # #student_model.cuda()

    # student_model = student_model().to(pytorch_device)
    # if args.local_rank >= 1:
    if distributed:
        student_model = DistributedDataParallel(
            student_model,
            device_ids=[pytorch_device],
            output_device=args.local_rank,
            find_unused_parameters=False  # True
        )
        teacher_model = DistributedDataParallel(
            teacher_model,
            device_ids=[pytorch_device],
            output_device=args.local_rank,
            find_unused_parameters=False  # True
        )

    # for weighted class loss
    weighted_class = False

    # for focal loss
    focal_loss = False  # True

    per_class_weight = None
    if focal_loss or weighted_class:
        # 20 class number of samples from training sample
        class_weights = semantic_kitti_class_weights
        per_class_weight = torch.from_numpy(class_weights).to(pytorch_device)

    if ssl:
        loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                       num_class=num_class, ignore_label=ignore_label,
                                                       weights=per_class_weight, ssl=True, fl=focal_loss)
    else:
        loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                       num_class=num_class, ignore_label=ignore_label,
                                                       weights=per_class_weight, fl=focal_loss)

    source_train_dataset_loader, source_val_dataset_loader, _, target_train_dataset_loader = data_builder.build(
        dataset_config,
        train_dataloader_config,
        val_dataloader_config,
        ssl_dataloader_config=ssl_dataloader_config,
        grid_size=grid_size,
        train_hypers=train_hypers)

    optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=train_hypers["learning_rate"])
    optimizer_student = optim.Adam(student_model.parameters(), lr=train_hypers["learning_rate"])

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer_student, max_lr=0.01,
    #                                                 steps_per_epoch=len(source_train_dataset_loader),
    #                                                 epochs=train_hypers["max_num_epochs"])

    # global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    global global_iter, best_val_miou, epoch
    print("|-------------------------Training started-----------------------------------------|")
    print(f"focal_loss:{focal_loss}, weighted_cross_entropy: {weighted_class}")

    # Define training mode and function
    trainer = Trainer(student_model=student_model,
                      teacher_model=teacher_model,
                      optimizer_teacher=optimizer_teacher,
                      optimizer_student=optimizer_student,
                      ckpt_dir=model_save_path,
                      unique_label=unique_label,
                      unique_label_str=unique_label_str,
                      lovasz_softmax=lovasz_softmax,
                      loss_func=loss_func,
                      ignore_label=ignore_label,
                      train_mode="ema",
                      ssl=ssl,
                      eval_frequency=1,
                      pytorch_device=pytorch_device,
                      warmup_epoch=5,
                      ema_frequency=2)

    # train and val model
    trainer.uda_fit(train_hypers["max_num_epochs"],
                    source_train_dataset_loader,
                    source_train_batch_size,
                    target_train_dataset_loader,
                    target_train_batch_size,
                    source_val_dataset_loader,
                    source_val_batch_size,
                    test_loader=None)

    # trainer.fit(train_hypers["max_num_epochs"],
    #             source_train_dataset_loader,
    #             source_train_batch_size,
    #             source_val_dataset_loader,
    #             source_val_batch_size,
    #             test_loader=None,
    #             ckpt_save_interval=5,
    #             lr_scheduler_each_iter=False)

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('-y', '--config_path', default='configs/data_config/da_kitti_poss/uda_poss_kitti_f2_2_time.yaml')
    parser.add_argument('-y', '--config_path', default='configs/data_config/da_kitti_usl/uda_usl_kitti_f2_2_time.yaml')
    # parser.add_argument('-y', '--config_path', default='configs/data_config/semantickitti/semantickitti_f3_3_s10.yaml')
    parser.add_argument('-g', '--mgpus', action='store_true', default=False)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
