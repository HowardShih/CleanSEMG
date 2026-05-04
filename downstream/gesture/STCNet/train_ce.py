# ${CLEANSEMG_ROOT}/downstream_tasks/stcnet/STCNet/train_ce.py
import pandas as pd
import copy
import math
import time
import os
import argparse
import pickle
import numpy as np
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from dataset import NinaDataset
from util import AverageMeter, AccuracyMeter
from util import save_model, get_data, get_model
from augmentations import GaussianNoise, MagnitudeWarping, WaveletDecomposition, Permute

# seed
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
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # model dataset
    parser.add_argument('--model', type=str, default='STCNet',
                        choices=['STCNet', 'baseline'], help='model')
    parser.add_argument('--stc', action='store_true')
    parser.add_argument('--encoder', type=str, default=None, help='path to encoder weights')
    parser.add_argument('--freeze', action='store_true', help='freeze encoder weights')
    parser.add_argument('--dataset', type=str, default='nina1',
                        choices=['nina1', 'nina2', 'nina4'], help='dataset')

    # augmentations
    parser.add_argument('--aug', action='store_true',
                        help='using augmentations')
    parser.add_argument('--prob', type=float, default=0.5,
                        help='probablity for augmentations')

    # kfold
    parser.add_argument('--kfold', type=int, default=-1,
                        help='if you want to use kfold val, choose 0~4')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()
    stc_cond = ''
    if opt.stc and opt.model == 'baseline':
        stc_cond = 'stc'

    opt.model_path = './save/CE/{}_{}_{}models'.format(opt.dataset, opt.model, stc_cond)
    opt.tb_path = './save/CE/{}_{}_{}tensorboard'.format(opt.dataset, opt.model, stc_cond)
    opt.pkl_path = './save/CE/{}_{}_{}pkl'.format(opt.dataset, opt.model, stc_cond)

    opt.model_name = 'lr_{}_decay_{}_bsz_{}_tri_{}'.format(
        opt.learning_rate, opt.weight_decay, opt.batch_size, opt.trial
    )

    if opt.kfold in range(5):
        opt.model_name = 'kfold{}_{}'.format(opt.kfold, opt.model_name)

    if opt.encoder is not None:
        enc_cfg = opt.encoder.split('/')[-2]
        opt.model_name = '{}_enc_{}'.format(opt.model_name, enc_cfg)

    if opt.cosine:
        opt.model_name = '{}_cos'.format(opt.model_name)

    if opt.aug:
        opt.model_name = '{}_aug_{}'.format(opt.model_name, opt.prob)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    os.makedirs(opt.tb_folder, exist_ok=True)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    opt.pkl_folder = os.path.join(opt.pkl_path, opt.model_name)
    os.makedirs(opt.pkl_folder, exist_ok=True)

    return opt


def set_loader(opt):
    train, test = get_data(opt.dataset, opt.kfold)

    test_dataset = NinaDataset(test, dataset=opt.dataset, model=opt.model)

    if opt.aug:
        train_transform = transforms.Compose([
            GaussianNoise(p=opt.prob),
            MagnitudeWarping(p=opt.prob),
            WaveletDecomposition(p=opt.prob),
            Permute(data=opt.dataset, model=opt.model)
        ])
        train_dataset = NinaDataset(train, dataset=opt.dataset, model=opt.model, transform=train_transform)
    else:
        train_dataset = NinaDataset(train, dataset=opt.dataset, model=opt.model)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def set_model(opt):
    """
    終極修正版 - 自動適應模型是否已被 DataParallel 包裝
    """
    criterion = torch.nn.CrossEntropyLoss()
    model = get_model(opt)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    if opt.encoder is not None:
        print(f"Loading encoder weights from {opt.encoder}")
        ckpt = torch.load(
            opt.encoder,
            map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )

        ckpt_sd = ckpt.get('model', ckpt)

        # 1) 只拿 encoder.* （不要用 'encoder' in k）
        enc_sd = {}
        for k, v in ckpt_sd.items():
            if k.startswith('encoder.'):
                kk = k[len('encoder.'):]     # remove encoder.
                if kk.startswith('module.'):
                    kk = kk[len('module.'):] # remove module.
                enc_sd[kk] = v

        # 2) CE 目標 encoder：先 unwrap DataParallel（如果有）
        target_encoder = model.encoder
        if isinstance(target_encoder, torch.nn.DataParallel):
            target_encoder = target_encoder.module

        # 3) 真的載入（strict=False 允許少量不匹配，但不該是 0 匹配）
        missing, unexpected = target_encoder.load_state_dict(enc_sd, strict=False)

        print("✓ Encoder load summary:")
        print(f"  - ckpt encoder params: {len(enc_sd)}")
        print(f"  - target encoder params: {len(target_encoder.state_dict())}")
        print(f"  - missing: {len(missing)}")
        print(f"  - unexpected: {len(unexpected)}")

        # 快速 sanity check：如果 unexpected == len(enc_sd)，代表還是 0 match
        if len(unexpected) == len(enc_sd):
            print("❌ 0 parameters matched. SAC encoder and CE encoder key-names are fully incompatible.")
            print("   Please print key samples to debug (see below).")

        if opt.freeze:
            for param in target_encoder.parameters():
                param.requires_grad = False
            print("✓ Encoder weights frozen")


    return model, criterion


def train_one_epoch(train_loader, model, criterion, optimizer):
    train_losses = AverageMeter()
    train_acc = AccuracyMeter()

    model.train()
    for _, (inputs, labels, _) in enumerate(train_loader):
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        features = model(inputs)
        loss = criterion(features, labels)

        train_losses.update(loss.item(), bsz)
        train_acc.update(features, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_losses.avg, train_acc.correct / train_acc.total


@torch.no_grad()
def validate(test_loader, model, criterion):
    val_losses = AverageMeter()
    val_acc = AccuracyMeter()

    model.eval()
    for _, (inputs, labels, _) in enumerate(test_loader):
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        features = model(inputs)
        loss = criterion(features, labels)

        val_losses.update(loss.item(), bsz)
        val_acc.update(features, labels)

    return val_losses.avg, val_acc.correct / val_acc.total


def set_optimizer(opt, model):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt.learning_rate,
        betas=(opt.beta1, opt.beta2),
        weight_decay=opt.weight_decay
    )
    return optimizer


def step_decay(epoch):
    drop = 0.1
    epochs_drop = 70.0
    lamb = math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lamb


def main():
    opt = parse_option()

    train_loader, test_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)

    scheduler = LambdaLR(optimizer, lr_lambda=step_decay)
    if opt.cosine:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=opt.epochs, eta_min=opt.learning_rate / 100
        )

    # TensorBoard (PyTorch native)
    writer = SummaryWriter(log_dir=opt.tb_folder)

    lrs = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_acc = 0.0
    best_model = None

    for epoch in range(1, opt.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(train_loader, model, criterion, optimizer)
        va_loss, va_acc = validate(test_loader, model, criterion)

        scheduler.step()
        t1 = time.time()

        print(
            'epoch {}, total time {:.2f} train_loss {:.2f} val_loss {:.2f} train_acc {:.2f} val_acc {:.2f}'.format(
                epoch, t1 - t0, tr_loss, va_loss, tr_acc * 100, va_acc * 100
            )
        )

        # TensorBoard scalars
        writer.add_scalar('train/loss', tr_loss, epoch)
        writer.add_scalar('val/loss', va_loss, epoch)
        writer.add_scalar('train/acc', tr_acc, epoch)
        writer.add_scalar('val/acc', va_acc, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # record for pkl
        lrs.append(optimizer.param_groups[0]['lr'])
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        if va_acc > best_acc:
            best_acc = va_acc
            best_model = copy.deepcopy(model)

    writer.close()

    print('best_acc: {}'.format(best_acc))
    save_file = os.path.join(opt.save_folder, 'best_model.pth')
    save_model(best_model, optimizer, opt, opt.epochs, save_file)

    save_pkl = os.path.join(opt.pkl_folder, 'figure.pkl')
    with open(save_pkl, 'wb') as f:
        pickle.dump({
            'lrs': lrs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_acc': best_acc
        }, f)


if __name__ == '__main__':
    main()