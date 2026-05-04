#!/usr/bin/env python3
#${CLEANSEMG_ROOT}/downstream_tasks/stcnet/STCNet/util.py
# -*- coding: utf-8 -*-
"""
STCNet Utility Functions (FIXED VERSION)

CRITICAL FIXES:
1. MetricsMeter: Corrected class counts (was using subject counts!)
   - nina1: 52 classes (not 27 subjects)
   - nina2: 49 classes (not 40 subjects)  
   - nina4: 52 classes (not 10 subjects)

2. PValueMeter: Same fix applied

Reference: STCNet Paper Table 1 (Dataset Details)
"""

import math
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from networks.STCNet import STCNetCE

import pandas as pd


# =============================================================================
# Dataset Configuration (from Paper Table 1)
# =============================================================================
DATASET_CFG = {
    'nina1': {
        'num_classes': 52,    # Number of gestures
        'num_subjects': 27,   # Number of subjects
        'num_channels': 10,   # Number of EMG channels
        'num_repetitions': 10,
        'trial_length': 500,
        'sample_rate': 100,   # Hz (already downsampled)
    },
    'nina2': {
        'num_classes': 49,    # Number of gestures
        'num_subjects': 40,   # Number of subjects
        'num_channels': 12,   # Number of EMG channels
        'num_repetitions': 6,
        'trial_length': 10000,
        'sample_rate': 2000,  # Hz (needs downsampling)
    },
    'nina4': {
        'num_classes': 52,    # Number of gestures
        'num_subjects': 10,   # Number of subjects
        'num_channels': 12,   # Number of EMG channels
        'num_repetitions': 6,
        'trial_length': 10000,
        'sample_rate': 2000,  # Hz (needs downsampling)
    },
}


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccuracyMeter(object):
    """Computes and stores accuracy"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()

    def compute(self):
        if self.total == 0:
            return 0.0
        return self.correct / self.total * 100


class MetricsMeter(object):
    """
    Computes precision, recall, F1-score, specificity, and balanced accuracy.
    
    FIXED: Now uses correct number of GESTURE CLASSES (not subjects!)
    
    From Paper Table 1:
    - nina1: 52 classes
    - nina2: 49 classes
    - nina4: 52 classes
    """
    def __init__(self, dataset):
        # FIXED: Use gesture class counts from paper Table 1
        if dataset not in DATASET_CFG:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        self.num_classes = DATASET_CFG[dataset]['num_classes']
        self.dataset = dataset
        self.reset()

    def reset(self):
        self.TP = torch.zeros(self.num_classes)
        self.FP = torch.zeros(self.num_classes)
        self.FN = torch.zeros(self.num_classes)
        self.TN = torch.zeros(self.num_classes)
        self.targets = []
        self.outputs = []

    def update(self, outputs, labels):
        outputs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        self.outputs.extend(outputs.detach().cpu().numpy())
        self.targets.extend(labels.detach().cpu().numpy())

        for i in range(self.num_classes):
            tp = ((predicted == i) & (labels == i)).sum().item()
            fp = ((predicted == i) & (labels != i)).sum().item()
            fn = ((predicted != i) & (labels == i)).sum().item()
            tn = ((predicted != i) & (labels != i)).sum().item()

            self.TP[i] += tp
            self.FP[i] += fp
            self.FN[i] += fn
            self.TN[i] += tn

    def compute_metrics(self):
        # Avoid division by zero
        eps = 1e-8
        
        precision = self.TP / (self.TP + self.FP + eps)
        recall = self.TP / (self.TP + self.FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        specificity = self.TN / (self.TN + self.FP + eps)
        balanced_accuracy = (recall + specificity) / 2

        # Handle NaN values
        precision = torch.nan_to_num(precision, nan=0.0)
        recall = torch.nan_to_num(recall, nan=0.0)
        f1 = torch.nan_to_num(f1, nan=0.0)
        specificity = torch.nan_to_num(specificity, nan=0.0)
        balanced_accuracy = torch.nan_to_num(balanced_accuracy, nan=0.0)

        return {
            "precision": precision.mean().item() * 100,
            "recall": recall.mean().item() * 100,
            "specificity": specificity.mean().item() * 100,
            "f1_score": f1.mean().item() * 100,
            "balanced_accuracy": balanced_accuracy.mean().item() * 100
        }


class PValueMeter(object):
    """
    Per-class accuracy meter.
    
    FIXED: Now uses correct number of GESTURE CLASSES (not subjects!)
    """
    def __init__(self, dataset):
        if dataset not in DATASET_CFG:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        self.num_classes = DATASET_CFG[dataset]['num_classes']
        self.reset()

    def reset(self):
        self.correct = torch.zeros(self.num_classes)
        self.total = torch.zeros(self.num_classes)

    def update(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)

        for i in range(self.num_classes):
            correct_ = ((predicted == i) & (labels == i)).sum().item()
            total_ = (labels == i).sum().item()

            self.correct[i] += correct_
            self.total[i] += total_

    def compute_metrics(self):
        # Avoid division by zero
        total_safe = self.total.clone()
        total_safe[total_safe == 0] = 1
        acc = self.correct / total_safe
        return acc


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


FOLD_CFG = {
    'nina1': ((22, 5), (22, 5), (22, 5), (21, 6), (21, 6)),
    'nina2': ((32, 8), (32, 8), (32, 8), (32, 8), (32, 8)),
    'nina4': ((8, 2), (8, 2), (8, 2), (8, 2), (8, 2))
}


def get_data(dataset, k=-1):
    """Load train/test data from PKL files"""
    if k not in range(5):
        train = pd.read_pickle(f'./pkl/train_{dataset}.pkl')
        test = pd.read_pickle(f'./pkl/test_{dataset}.pkl')
        return train, test
    
    data = pd.read_pickle(f'./pkl/{dataset}_fold.pkl')
    train = data[data['fold'] != k]
    test = data[data['fold'] == k]

    train.loc[:, 'subject'] = train['subject'].map(
        pd.Series(index=train['subject'].unique(), data=range(FOLD_CFG[dataset][k][0]))
    )
    test.loc[:, 'subject'] = test['subject'].map(
        pd.Series(index=test['subject'].unique(), data=range(FOLD_CFG[dataset][k][1]))
    )

    return train, test


def get_model(opt):
    """Get STCNet model for the specified dataset"""
    return STCNetCE(data=opt.dataset)


# =============================================================================
# Diagnostic Functions
# =============================================================================
def verify_dataset_config():
    """Print dataset configuration for verification"""
    print("=" * 60)
    print("STCNet Dataset Configuration (from Paper Table 1)")
    print("=" * 60)
    for ds, cfg in DATASET_CFG.items():
        print(f"\n{ds.upper()}:")
        for key, val in cfg.items():
            print(f"  {key}: {val}")
    print("=" * 60)


def verify_pkl_contents(dataset):
    """Verify PKL file contents match expected format"""
    print(f"\n[Verifying PKL files for {dataset}]")
    
    try:
        train = pd.read_pickle(f'./pkl/train_{dataset}.pkl')
        test = pd.read_pickle(f'./pkl/test_{dataset}.pkl')
        
        print(f"\nTrain PKL:")
        print(f"  Samples: {len(train)}")
        print(f"  Columns: {list(train.columns)}")
        if 'stimulus' in train.columns:
            print(f"  Unique classes: {train['stimulus'].nunique()}")
            print(f"  Class range: [{train['stimulus'].min()}, {train['stimulus'].max()}]")
        if 'subject' in train.columns:
            print(f"  Unique subjects: {train['subject'].nunique()}")
        
        print(f"\nTest PKL:")
        print(f"  Samples: {len(test)}")
        if 'stimulus' in test.columns:
            print(f"  Unique classes: {test['stimulus'].nunique()}")
        
        # Check expected counts from paper
        expected = {
            'nina1': (9828, 4212),
            'nina2': (7840, 3920),
            'nina4': (2080, 1040),
        }
        
        if dataset in expected:
            exp_train, exp_test = expected[dataset]
            train_match = "✓" if len(train) == exp_train else f"❌ (expected {exp_train})"
            test_match = "✓" if len(test) == exp_test else f"❌ (expected {exp_test})"
            print(f"\nSample count check:")
            print(f"  Train: {len(train)} {train_match}")
            print(f"  Test: {len(test)} {test_match}")
            
    except FileNotFoundError as e:
        print(f"  PKL file not found: {e}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == '__main__':
    verify_dataset_config()
    
    for ds in ['nina1', 'nina2', 'nina4']:
        try:
            verify_pkl_contents(ds)
        except:
            pass