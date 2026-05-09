# ${CLEANSEMG_ROOT}/downstream_tasks/stcnet/STCNet/train_sac.py

import argparse
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from augmentations import GaussianNoise, MagnitudeWarping, Permute, WaveletDecomposition
from dataset import NinaDataset
from losses import SACLoss
from networks.STCNet import STCNetSAC
from util import AverageMeter, TwoCropTransform
from util import adjust_learning_rate, get_data, save_model, set_optimizer

# Reproducibility settings
seed = 42
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_option():
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="print frequency",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="number of data loading workers",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of training epochs",
    )

    # Optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="learning rate",
    )
    parser.add_argument(
        "--lr_decay_epochs",
        type=str,
        default="70,140",
        help="epochs at which to decay the learning rate",
    )
    parser.add_argument(
        "--lr_decay_rate",
        type=float,
        default=0.1,
        help="learning rate decay factor",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="weight decay",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum",
    )

    # Model and dataset
    parser.add_argument(
        "--model",
        type=str,
        default="STCNet",
        help="model architecture",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nina1",
        choices=["nina1", "nina2", "nina4"],
        help="dataset name",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=None,
        help="path to custom dataset",
    )

    # Contrastive objective
    parser.add_argument(
        "--method",
        type=str,
        default="SupCon",
        choices=["SupCon", "SimCLR"],
        help="contrastive learning method",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="weight for the subject-aware contrastive term",
    )

    # Cross-validation
    parser.add_argument(
        "--kfold",
        type=int,
        default=-1,
        help="k-fold split index; use 0-4 to enable k-fold validation",
    )

    # Data augmentation
    parser.add_argument(
        "--prob",
        type=float,
        default=0.5,
        help="augmentation probability",
    )

    # Loss temperature
    parser.add_argument(
        "--temp",
        type=float,
        default=0.07,
        help="temperature for the contrastive loss",
    )

    # Training options
    parser.add_argument(
        "--cosine",
        action="store_true",
        help="enable cosine annealing learning-rate schedule",
    )
    parser.add_argument(
        "--syncBN",
        action="store_true",
        help="enable synchronized batch normalization",
    )
    parser.add_argument(
        "--warm",
        action="store_true",
        help="enable warm-up for large-batch training",
    )
    parser.add_argument(
        "--trial",
        type=str,
        default="0",
        help="run identifier",
    )

    opt = parser.parse_args()

    if opt.data_folder is None:
        opt.data_folder = "./datasets/"

    opt.model_path = "./save/SAC/{}_models".format(opt.dataset)
    opt.tb_path = "./save/SAC/{}_tensorboard".format(opt.dataset)
    opt.pkl_path = "./save/SAC/{}_pkl".format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = []
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = "lr_{}_decay_{}_bsz_{}_temp_{}_tri_{}_gamma_{}".format(
        opt.learning_rate,
        opt.weight_decay,
        opt.batch_size,
        opt.temp,
        opt.trial,
        opt.gamma,
    )

    if opt.kfold in range(5):
        opt.model_name = "kfold{}_{}".format(opt.kfold, opt.model_name)

    if opt.cosine:
        opt.model_name = "{}_cos".format(opt.model_name)

    if opt.batch_size > 256:
        opt.warm = True

    if opt.warm:
        opt.model_name = "{}_warm".format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10

        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)
            ) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    os.makedirs(opt.tb_folder, exist_ok=True)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    opt.pkl_folder = os.path.join(opt.pkl_path, opt.model_name)
    os.makedirs(opt.pkl_folder, exist_ok=True)

    return opt


def set_loader(opt):
    train, _ = get_data(opt.dataset, opt.kfold)

    train_transform = transforms.Compose(
        [
            GaussianNoise(p=opt.prob),
            MagnitudeWarping(p=opt.prob),
            WaveletDecomposition(p=opt.prob),
            Permute(data=opt.dataset, model=opt.model),
        ]
    )

    train_dataset = NinaDataset(
        train,
        dataset=opt.dataset,
        transform=TwoCropTransform(train_transform),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    return train_loader


def set_model(opt):
    criterion = SACLoss(temperature=(opt.temp, opt.temp), gamma=opt.gamma)
    model = STCNetSAC(dataset=opt.dataset)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)

        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    losses = AverageMeter()

    for idx, (inputs, labels, subjects) in enumerate(train_loader):
        # TwoCropTransform returns two augmented views of the same sample.
        inputs = torch.cat([inputs[0], inputs[1]], dim=0)

        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            subjects = subjects.cuda(non_blocking=True)

        bsz = labels.shape[0]

        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        labels_features, subjects_features = model(inputs)

        f1, f2 = torch.split(labels_features, [bsz, bsz], dim=0)
        labels_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        f3, f4 = torch.split(subjects_features, [bsz, bsz], dim=0)
        subjects_features = torch.cat([f3.unsqueeze(1), f4.unsqueeze(1)], dim=1)

        loss = criterion(labels_features, subjects_features, labels, subjects)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx + 1) % opt.print_freq == 0:
            print(
                "Train: [{0}][{1}/{2}]\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    loss=losses,
                )
            )
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    train_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)

    writer = SummaryWriter(log_dir=opt.tb_folder)

    lrs = []
    losses = []

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        t0 = time.time()
        loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, opt)
        t1 = time.time()

        print("epoch {}, total time {:.2f}".format(epoch, t1 - t0))

        writer.add_scalar("train/loss", loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        lrs.append(optimizer.param_groups[0]["lr"])
        losses.append(loss)

    writer.close()

    save_file = os.path.join(opt.save_folder, "last.pth")
    save_model(model, optimizer, opt, opt.epochs, save_file)

    save_pkl = os.path.join(opt.pkl_folder, "figure.pkl")
    with open(save_pkl, "wb") as f:
        pickle.dump(
            {
                "lrs": lrs,
                "losses": losses,
            },
            f,
        )


if __name__ == "__main__":
    main()