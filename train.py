# -*- coding:utf-8 -*-
# author: Awet
# @file: train_cylinder_asym_wod.py

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

    model_load_path = train_hypers['teacher_model_path']
    model_save_path = train_hypers['teacher_model_path']

    SemKITTI_label_name = get_label_name(dataset_config["label_mapping"])
    # NB: no ignored class
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config).to(pytorch_device)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model, map_location=pytorch_device)

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

    train_dataset_loader, val_dataset_loader, _, _ = data_builder.build(dataset_config,
                                                                        train_dataloader_config,
                                                                        val_dataloader_config,
                                                                        ssl_dataloader_config=ssl_dataloader_config,
                                                                        grid_size=grid_size,
                                                                        train_hypers=train_hypers)

    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataset_loader),
                                                    epochs=train_hypers["max_num_epochs"])

    my_model.train()
    # global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    global global_iter, best_val_miou, epoch
    print("|-------------------------Training started-----------------------------------------|")
    print(f"focal_loss:{focal_loss}, weighted_cross_entropy: {weighted_class}")

    while epoch < train_hypers['max_num_epochs']:
        print(f"epoch: {epoch}")
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(15)

        # lr_scheduler.step(epoch)

        def validating(hist_list, val_loss_list, val_vox_label, val_grid, val_pt_labs, val_pt_fea, ref_st_idx=None,
                       ref_end_idx=None, lcw=None):

            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

            predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
            # aux_loss = loss_fun(aux_outputs, point_label_tensor)

            inp = val_label_tensor.size(0)

            # TODO: check if this is correctly implemented
            # hack for batch_size mismatch with the number of training example
            predict_labels = predict_labels[:inp, :, :, :, :]

            if ssl:
                lcw_tensor = torch.FloatTensor(lcw).to(pytorch_device)
                loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                      lcw=lcw_tensor) + loss_func(predict_labels.detach(),
                                                                  val_label_tensor, lcw=lcw_tensor)
            else:
                loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(),
                                      val_label_tensor) + loss_func(predict_labels.detach(), val_label_tensor)

            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()
            for count, i_val_grid in enumerate(val_grid):
                hist_list.append(fast_hist_crop(predict_labels[
                                                    count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                    val_grid[count][:, 2]], val_pt_labs[count],
                                                unique_label))
            val_loss_list.append(loss.detach().cpu().numpy())

            return hist_list, val_loss_list

        # if global_iter % check_iter == 0 and epoch >= 1:
        if epoch >= 1:
            my_model.eval()
            hist_list = []
            val_loss_list = []
            with torch.no_grad():
                my_model.eval()

                # Validation with multi-frames and ssl:
                # if past_frame > 0 and train_hypers['ssl']:
                for i_iter_val, (
                        _, vox_label, grid, pt_labs, pt_fea, ref_st_idx, ref_end_idx, val_lcw) in enumerate(
                    val_dataset_loader):
                    # call the validation and inference with
                    hist_list, val_loss_list = validating(hist_list, val_loss_list, vox_label, grid, pt_labs,
                                                          pt_fea, ref_st_idx=ref_st_idx,
                                                          ref_end_idx=ref_end_idx,
                                                          lcw=val_lcw)

                print(f"--------------- epoch: {epoch} ----------------")

                iou = per_class_iu(sum(hist_list))
                print('Validation per class iou: ')
                for class_name, class_iou in zip(unique_label_str, iou):
                    print('%s : %.2f%%' % (class_name, class_iou * 100))
                val_miou = np.nanmean(iou) * 100
                # del val_vox_label, val_grid, val_pt_fea

                # save model if performance is improved
                if best_val_miou < val_miou:
                    best_val_miou = val_miou
                    torch.save(my_model.state_dict(), model_save_path)

                print('Current val miou is %.3f while the best val miou is %.3f' %
                      (val_miou, best_val_miou))
                print('Current val loss is %.3f' %
                      (np.mean(val_loss_list)))

        def training(i_iter_train, train_vox_label, train_grid, pt_labels, train_pt_fea, ref_st_idx=None,
                     ref_end_idx=None, lcw=None):
            global global_iter, best_val_miou, epoch

            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)

            # forward + backward + optimize
            outputs = my_model(train_pt_fea_ten, train_vox_ten, train_batch_size)
            inp = point_label_tensor.size(0)
            # print(f"outputs.size() : {outputs.size()}")
            # TODO: check if this is correctly implemented
            # hack for batch_size mismatch with the number of training example
            outputs = outputs[:inp, :, :, :, :]
            ################################

            if ssl:
                lcw_tensor = torch.FloatTensor(lcw).to(pytorch_device)
                loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=ignore_label,
                                      lcw=lcw_tensor) + loss_func(
                    outputs, point_label_tensor, lcw=lcw_tensor)
            else:
                loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor,
                                      ignore=ignore_label) + loss_func(
                    outputs, point_label_tensor)

            # TODO: check --> to mitigate only one element tensors can be converted to Python scalars
            # loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Uncomment to use the learning rate scheduler
            # scheduler.step()

            loss_list.append(loss.item())

            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter_train, np.mean(loss_list)))
                else:
                    print('loss error')

            global_iter += 1

            if global_iter % 100 == 0:
                pbar.update(100)

            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter_train, np.mean(loss_list)))
                else:
                    print('loss error')

        my_model.train()
        # training with multi-frames and ssl:
        for i_iter_train, (_, vox_label, grid, pt_labs, pt_fea, ref_st_idx, ref_end_idx, lcw) in enumerate(
                train_dataset_loader):
            # call the validation and inference with
            training(i_iter_train, vox_label, grid, pt_labs, pt_fea, ref_st_idx=ref_st_idx, ref_end_idx=ref_end_idx,
                     lcw=lcw)

        pbar.close()
        epoch += 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='configs/data_config/da_livoxsim_livoxreal/uda_livoxsim_livoxreal_f0_0_time.yaml')
    # parser.add_argument('-y', '--config_path', default='configs/data_config/wod/wod_f0_0_intensity_beam32.yaml')
    # parser.add_argument('-y', '--config_path', default='configs/semantickitti/semantickitti_f3_3_s10.yaml')
    parser.add_argument('-g', '--mgpus', action='store_true', default=False)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
