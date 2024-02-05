#!/usr/bin/env python
# coding=utf-8

"""
@author: Richard Huang
@license: WHU
@contact: 2539444133@qq.com
@file: train.py
@date: 22/05/02 14:44
@desc: 
"""
"""Evaluate pre-trained LipForensics model on various face forgery datasets"""

import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop
from tqdm import tqdm
import numpy as np
import random
import os
import time

from data.transforms import NormalizeVideo, ToTensorVideo
from data.dataset_clips import ForensicsClips, DFDCClips
from data.samplers import ConsecutiveClipSampler
from models.spatiotemporal_net import get_model
from utils import get_files_from_split
from utils import get_save_folder
from utils import get_logger
from utils import CheckpointSaver, AverageMeter
from utils import get_optimizer
from utils import CosineScheduler
from utils import load_model, showLR, update_logger_batch
from utils import mixup_data, mixup_criterion

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description="DeepFake detector evaluation")
    parser.add_argument(
        "--dataset", help="Dataset to evaluate on", type=str, choices={"FaceForensics++", "DFDC"}, default="DFDC")
    parser.add_argument(
        "--compression",
        help="Video compression level for FaceForensics++", type=str, choices=["c0", "c23", "c40"], default="c23"
    )
    parser.add_argument("--grayscale", dest="grayscale", action="store_true")  # 灰度图
    parser.add_argument("--rgb", dest="grayscale", action="store_false")  # 彩图
    parser.set_defaults(grayscale=True)
    parser.add_argument("--frames_per_clip", default=29, type=int)  # 每个clip的帧数
    parser.add_argument("--batch_size", default=4, type=int)  # batch_size
    parser.add_argument("--device", help="Device to put tensors on", type=str, default="cuda:0")
    parser.add_argument("--num_workers", default=4, type=int)  # 线程数
    parser.add_argument('--model-path', type=str, help='Pretrained model pathname')
    parser.add_argument(                                        # 唇读预训练模型
        "--weights_forgery_path", help="Path to pretrained weights for forgery detection", type=str,
        default="./models/lipforensics_ff.pth"
    )
    parser.add_argument(
        "--train_path", help="Path to train splits", type=str,
        default="./data/datasets/DFDC/train.json"
    )
    parser.add_argument(
        "--test_path", help="Path to test splits", type=str,
        default="./data/datasets/DFDC/test.json"
    )
    parser.add_argument(
        "--val_path", help="Path to val splits", type=str,
        default="attack/xception/test.json"
    )
    parser.add_argument('--allow-size-mismatch', default=False, action='store_true')
    parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--init-epoch', default=0, type=int, help='epoch to start at')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--interval', default=50, type=int, help='display interval')
    parser.add_argument('--logging-dir', type=str, default='./train_logs',
                        help='path to the directory in which to save the log file')
    parser.add_argument('--training-mode', default='Both', help='visual, audio, Both')
    parser.add_argument('--model_mode', default='train', help='train, eval')

    args = parser.parse_args()

    return args


def train(model, dset_loader, criterion, epoch, optimizer, logger, args):
    data_time = AverageMeter()
    batch_time = AverageMeter()

    lr = showLR(optimizer)

    logger.info('-' * 10)
    logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
    logger.info('Current learning rate: {}'.format(lr))

    model.train()
    running_loss = 0.
    running_corrects = 0.
    running_all = 0.

    end = time.time()
    for batch_idx, data in enumerate(tqdm(dset_loader)):
        images, audios, labels, video_indices = data
        images = images.to(args.device, dtype=torch.float)
        labels = labels.to(args.device, dtype=torch.int64)
        audios = audios.to(args.device, dtype=torch.float)
        data_time.update(time.time() - end)

        # --
        images, labels_a, labels_b, lam = mixup_data(images, labels, args.alpha)
        labels_a, labels_b = labels_a.to(args.device), labels_b.to(args.device)

        optimizer.zero_grad()
        logits = model(images, audios, lengths=[args.frames_per_clip] * images.shape[0])

        loss_func = mixup_criterion(labels_a, labels_b, lam)
        loss = loss_func(criterion, logits)

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # -- compute running performance
        _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
        running_loss += loss.item() * images.size(0)
        running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(
            labels_b.view_as(predicted)).sum().item()
        running_all += images.size(0)
        # -- log intermediate results
        if batch_idx % args.interval == 0 or (batch_idx == len(dset_loader) - 1):
            update_logger_batch(args, logger, dset_loader, batch_idx, running_loss, running_corrects, running_all,
                                batch_time, data_time)

    return model


def evaluate(model, dset_loader, criterion, args):
    model.eval()

    running_loss = 0.
    running_corrects = 0.

    with torch.no_grad():
        for i, data in enumerate(tqdm(dset_loader)):
            images, audios, labels, video_indices = data
            images = images.to(args.device, dtype=torch.float)
            labels = labels.to(args.device, dtype=torch.int64)
            audios = audios.to(args.device, dtype=torch.float)
            logits = model(images, audios, lengths=[args.frames_per_clip] * images.shape[0])
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(labels.view_as(preds)).sum().item()

            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

    print('{} in total\tCR: {}'.format(len(dset_loader.dataset), running_corrects / len(dset_loader.dataset)))
    return running_corrects / len(dset_loader.dataset), running_loss / len(dset_loader.dataset)


def main():
    args = parse_args()
    save_path = get_save_folder(args)
    print("Model and log being saved in: {}".format(save_path))
    logger = get_logger(args, save_path)
    ckpt_saver = CheckpointSaver(save_path)
    model = get_model(weights_forgery_path=args.weights_forgery_path, device=args.device, mode=args.training_mode)

    # Get dataset
    transform = Compose(
        [ToTensorVideo(), CenterCrop((88, 88)), NormalizeVideo((0.421,), (0.165,))]
    )
    if args.dataset in [
        "FaceForensics++",
        "Deepfakes",
        "FaceSwap",
        "Face2Face",
        "NeuralTextures",
        "FaceShifter",
        "DeeperForensics"
    ]:
        if args.dataset == "FaceForensics++":
            # fake_types = ("Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures")
            fake_types = ["Deepfakes"]
        else:
            fake_types = (args.dataset,)

        data_path = {'train': args.train_path, 'val': args.val_path, 'test': args.test_path}
        data_split = {partition:
                          pd.read_json(data_path[partition], dtype=False)
                      for partition in ['train', 'val', 'test']
                      }
        file = {partition:
                    get_files_from_split(data_split[partition])
                for partition in ['train', 'val', 'test']
                }
        dataset = {partition: ForensicsClips(file[partition][0], file[partition][1], args.frames_per_clip,
                                             fakes=fake_types, compression=args.compression, grayscale=args.grayscale,
                                             transform=transform, max_frames_per_video=270) for partition in
                   ['train', 'val', 'test']}
        # Get sampler that splits video into non-overlapping clips
        sampler = ConsecutiveClipSampler(dataset['train'].clips_per_video)
        loader = {x:
                      DataLoader(dataset[x], batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
                  for x in ['train', 'val', 'test']
                  }
    else:
        data_path = {'train': args.train_path, 'val': args.val_path, 'test': args.test_path}
        dataset = {partition: DFDCClips(args.frames_per_clip, data_path[partition], args.grayscale, transform) for
                   partition in
                   ['train', 'val', 'test']}

        loader = {x:
                      DataLoader(dataset[x], batch_size=args.batch_size,
                                 num_workers=args.num_workers, shuffle=True)
                  for x in ['train', 'val', 'test']
                  }

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, optim_policies=model.parameters())
    scheduler = CosineScheduler(args.lr, args.epochs)

    if args.model_path:
        assert args.model_path.endswith('.tar') and os.path.isfile(args.model_path), \
            "'.tar' model path does not exist. Path input: {}".format(args.model_path)
        # resume from checkpoint
        if args.init_epoch > 0:
            model, optimizer, epoch_idx, ckpt_dict = load_model(args.model_path, model, optimizer)
            args.init_epoch = epoch_idx
            ckpt_saver.set_best_from_ckpt(ckpt_dict)
            logger.info('Model and states have been successfully loaded from {}'.format(args.model_path))
        # init from trained model
        else:
            model = load_model(args.model_path, model, allow_size_mismatch=args.allow_size_mismatch)
            logger.info('Model has been successfully loaded from {}'.format(args.model_path))

        # -- fix learning rate after loading the checkpoint (latency)
    if args.model_path and args.init_epoch > 0:
        scheduler.adjust_lr(optimizer, args.init_epoch - 1)

    if args.model_mode == 'train':
        epoch = args.init_epoch
        while epoch < args.epochs:
            model = train(model, loader['train'], criterion, epoch, optimizer, logger, args)
            acc_avg_val, loss_avg_val = evaluate(model, loader['val'], criterion, args)
            logger.info(
                '{} Epoch:\t{:2}\tLoss val: {:.4f}\tAcc val:{:.4f}, LR: {}'.format('val', epoch, loss_avg_val, acc_avg_val,
                                                                                   showLR(optimizer)))
            # -- save checkpoint
            save_dict = {
                'epoch_idx': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            ckpt_saver.save(save_dict, acc_avg_val)
            scheduler.adjust_lr(optimizer, epoch)
            epoch += 1
    elif args.model_mode == 'eval':
        acc_avg_val, loss_avg_val = evaluate(model, loader['val'], criterion, args)
        logger.info(
            '{}\tLoss val: {:.4f}\tAcc val:{:.4f}, LR: {}'.format('val', loss_avg_val, acc_avg_val,
                                                                               showLR(optimizer)))


if __name__ == "__main__":
    main()
