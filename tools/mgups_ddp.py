#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# from transformers import BertForMaskedLM

SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 3


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class YourDataset(Dataset):

    def __init__(self):
        pass


def main():
    parser = ArgumentParser('DDP usage example')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N',
                        help='Local process rank.')  # you need this argument in your scripts for DDP to work
    args = parser.parse_args()

    # keep track of whether the current process is the `master` process (totally optional, but I find it useful for data laoding, logging, etc.)
    args.is_master = args.local_rank == 0

    # set the device
    device = torch.cuda.device(args.local_rank)

    # initialize PyTorch distributed using environment variables (you could also do this more explicitly by specifying `rank` and `world_size`, but I find using environment variables makes it so that you can easily use the same script on different machines)
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

    # set the seed for all GPUs (also make sure to set the seed for random, numpy, etc.)
    torch.cuda.manual_seed_all(SEED)

    # initialize your model (BERT in this example)
    model = ConvNet()

    # send your model to GPU
    model = model.to(device)

    # initialize distributed data parallel (DDP)
    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank
    )

    # initialize your dataset
    dataset = YourDataset()

    # initialize the DistributedSampler
    sampler = DistributedSampler(dataset)

    # initialize the dataloader
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=BATCH_SIZE
    )

    # start your training!
    for epoch in range(NUM_EPOCHS):
        # put model in train mode
        model.train()

        # let all processes sync up before starting with a new epoch of training
        dist.barrier()

        for step, batch in enumerate(dataloader):
            # send batch to device
            batch = tuple(t.to(device) for t in batch)

            # forward pass
            outputs = model(*batch)

            # compute loss
            loss = outputs[0]

            # etc.


if __name__ == '__main__':
    main()
