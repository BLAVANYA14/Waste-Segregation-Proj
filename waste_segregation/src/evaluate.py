"""Evaluate a trained model on the test split and compute accuracy, precision, recall, and confusion matrix.

Usage:
    python src/evaluate.py --data_dir data/processed --ckpt models/best_efficientnet_b0.pth --model efficientnet_b0
"""
import argparse
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
import timm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import WasteDataset
from transforms import get_transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data/processed')
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--model', type=str, default='efficientnet_b0')
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out_dir', type=str, default='results')
    return p.parse_args()


def load_model(model_name, num_classes, ckpt_path, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    ck = torch.load(ckpt_path, map_location=device)
    # support checkpoint dict with 'model_state_dict' or raw state_dict
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        state = ck['model_state_dict']
    else:
        state = ck
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate(model, loader, device):
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            ps.extend(preds.tolist())
            ys.extend(yb.numpy().tolist())
    ys = np.array(ys)
    ps = np.array(ps)
    return ys, ps


def plot_confusion(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load test dataset
    test_ds = WasteDataset(args.data_dir, split='test', transform=get_transforms('torchvision', 'val', args.img_size))
    labels = test_ds.classes
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # build & load model
    model = load_model(args.model, num_classes=len(labels), ckpt_path=args.ckpt, device=device)

    y_true, y_pred = evaluate(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(len(labels)))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    print(f'Overall accuracy: {acc:.4f}')
    print('Per-class precision, recall, f1:')
    for i, lbl in enumerate(labels):
        print(f'  {lbl:12s}  precision={prec[i]:.4f}  recall={rec[i]:.4f}  f1={f1[i]:.4f}')
    print('\nMacro-averaged:')
    print(f'  precision={prec_macro:.4f}  recall={rec_macro:.4f}  f1={f1_macro:.4f}')

    # classification report
    print('\nFull classification report:')
    print(classification_report(y_true, y_pred, target_names=labels))

    # confusion matrix
    cm_path = out_dir / 'confusion_matrix.png'
    plot_confusion(y_true, y_pred, labels, cm_path)
    print('Saved confusion matrix to', cm_path)


if __name__ == '__main__':
    main()
