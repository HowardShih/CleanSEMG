#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ${CLEANSEMG_ROOT}/downstream_tasks/stcnet/STCNet/test.py
"""
STCNet Test Script (CSV version) - STRICTLY aligned with original test.py

Aligned behaviors:
1) Overall: AccuracyMeter + MetricsMeter(dataset)  (same as original)
2) Per-subject: accuracy only (same as original)
3) Inter-subject score: std of per-subject accuracies (same as original)

Extras:
- Save overall + per-subject CSV
- Record dataset/mode/model_path/timestamp for reproducibility
"""

import os
import argparse
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from dataset import NinaDataset
from util import AccuracyMeter, MetricsMeter, get_data, get_model


def parse_option():
    parser = argparse.ArgumentParser(description='STCNet Model Evaluation (CSV, aligned with original)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--dataset', type=str, required=True, choices=['nina1', 'nina2', 'nina4'], help='Dataset type')
    parser.add_argument('--model', type=str, default='STCNet', choices=['baseline', 'STCNet'], help='Model type')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Num workers')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Directory to save CSV results')
    parser.add_argument('--mode', type=str, default='unknown',
                        choices=['baseline', 'noisy', 'denoised', 'unknown'],
                        help='Preprocessing mode (record keeping only)')
    return parser.parse_args()


def load_model(opt):
    model = get_model(opt)
    ckpt = torch.load(opt.model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(ckpt['model'])

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    return model


def evaluate_overall(model, test_loader, dataset):
    """Same as original evaluate_model(): overall accuracy + overall MetricsMeter."""
    metrics_meter = MetricsMeter(dataset)
    acc_meter = AccuracyMeter()

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            outputs = model(inputs)
            metrics_meter.update(outputs, labels)
            acc_meter.update(outputs, labels)

    overall_acc = acc_meter.compute()
    overall_metrics = metrics_meter.compute_metrics()
    return overall_acc, overall_metrics


def evaluate_by_subject_accuracy_only(model, test_loader):
    """
    Same as original evaluate_by_subjects(): per-subject accuracy only.
    Returns:
      acc_by_subjects: dict {subject_id: accuracy%}
      stats_by_subjects: dict {subject_id: {'correct': int, 'total': int}}
    """
    stats = {}

    with torch.no_grad():
        for inputs, labels, subjects in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i, subject in enumerate(subjects):
                sid = int(subject.item())
                if sid not in stats:
                    stats[sid] = {'correct': 0, 'total': 0}

                if int(preds[i].item()) == int(labels[i].item()):
                    stats[sid]['correct'] += 1
                stats[sid]['total'] += 1

    acc = {}
    for sid, d in stats.items():
        acc[sid] = (d['correct'] / d['total'] * 100.0) if d['total'] > 0 else 0.0

    return acc, stats


def save_csv(opt, overall_acc, overall_metrics, acc_by_subjects, stats_by_subjects, inter_subject_score):
    os.makedirs(opt.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    overall_csv = os.path.join(opt.output_dir, f'{opt.dataset}_{opt.mode}_overall_{timestamp}.csv')
    overall_df = pd.DataFrame([{
        'dataset': opt.dataset,
        'mode': opt.mode,
        'model_path': opt.model_path,
        'accuracy': overall_acc,
        'precision': overall_metrics['precision'],
        'recall': overall_metrics['recall'],
        'f1_score': overall_metrics['f1_score'],
        'specificity': overall_metrics['specificity'],
        'balanced_accuracy': overall_metrics['balanced_accuracy'],
        'inter_subject_score': inter_subject_score,
        'timestamp': timestamp
    }])
    overall_df.to_csv(overall_csv, index=False)

    subject_csv = os.path.join(opt.output_dir, f'{opt.dataset}_{opt.mode}_subjects_{timestamp}.csv')
    rows = []
    for sid in sorted(acc_by_subjects.keys()):
        rows.append({
            'dataset': opt.dataset,
            'mode': opt.mode,
            'subject_id': sid,
            'accuracy': acc_by_subjects[sid],
            'correct': stats_by_subjects[sid]['correct'],
            'total': stats_by_subjects[sid]['total'],
            'timestamp': timestamp
        })
    subject_df = pd.DataFrame(rows)
    subject_df.to_csv(subject_csv, index=False)

    return overall_csv, subject_csv


def main():
    opt = parse_option()

    _, test = get_data(opt.dataset, -1)
    test_dataset = NinaDataset(test, dataset=opt.dataset, model=opt.model)
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    print("=" * 70)
    print("STCNet Evaluation (aligned with original test.py)")
    print("=" * 70)
    print(f"Dataset:      {opt.dataset}")
    print(f"Mode:         {opt.mode}")
    print(f"Model:        {opt.model}")
    print(f"Model path:   {opt.model_path}")
    print(f"Test samples: {len(test_dataset)}")
    print("=" * 70)

    model = load_model(opt)

    overall_acc, overall_metrics = evaluate_overall(model, test_loader, opt.dataset)
    acc_by_subjects, stats_by_subjects = evaluate_by_subject_accuracy_only(model, test_loader)
    inter_subject_score = float(np.std(list(acc_by_subjects.values())))

    print("\nAccuracy:", overall_acc)
    print("Metrics:", overall_metrics)
    print("\nAccuracy across subjects:")
    print(acc_by_subjects)
    print("inter-subject score:")
    print(inter_subject_score)

    overall_csv, subject_csv = save_csv(
        opt, overall_acc, overall_metrics, acc_by_subjects, stats_by_subjects, inter_subject_score
    )

    print("\n✓ Saved CSV:")
    print("  Overall :", overall_csv)
    print("  Subject :", subject_csv)


if __name__ == '__main__':
    main()