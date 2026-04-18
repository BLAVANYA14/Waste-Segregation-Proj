"""Train a waste classification model using transfer learning (timm).

Usage example:
    python src/train_transfer.py \
      --data_dir data/processed --model efficientnet_b0 --img_size 224 \
      --batch_size 32 --epochs 12 --freeze_epochs 3 --out_dir models

This script:
- builds a timm model with pretrained weights and `num_classes` set to dataset classes
- freezes backbone and trains classifier head for `freeze_epochs`
- then unfreezes all layers and fine-tunes for remaining epochs
"""
import argparse
import os
from pathlib import Path
import time
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm

from dataset import WasteDataset
from transforms import get_transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data/processed')
    p.add_argument('--model', type=str, default='efficientnet_b0')
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--freeze_epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out_dir', type=str, default='models')
    return p.parse_args()


def compute_class_weights(dataset):
    """Compute class weights based on inverse frequency."""
    # Count samples per class
    class_counts = Counter(dataset.labels)
    num_classes = len(dataset.classes)
    total_samples = len(dataset)
    
    # Compute inverse frequency weights
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)  # avoid division by zero
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    weights = np.array(weights, dtype=np.float32)
    # Normalize so minimum weight is 1.0
    weights = weights / weights.min()
    
    return torch.tensor(weights)


def build_model(name, num_classes, pretrained=True):
    # timm will create a model and replace the classifier head when num_classes provided
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model


def set_requires_grad(module, req):
    for p in module.parameters():
        p.requires_grad = req


def get_head_params(model):
    # common classifier attribute names
    for attr in ('classifier', 'fc', 'head', 'head.fc'):
        parts = attr.split('.')
        cur = model
        found = True
        try:
            for part in parts:
                cur = getattr(cur, part)
        except Exception:
            found = False
        if found:
            return list(cur.parameters())
    # fallback: parameters that require grad (should be none if frozen)
    return [p for p in model.parameters() if p.requires_grad]


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0


def train():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # datasets
    train_ds = WasteDataset(args.data_dir, split='train', transform=get_transforms('torchvision', 'train', args.img_size))
    val_ds = WasteDataset(args.data_dir, split='val', transform=get_transforms('torchvision', 'val', args.img_size))

    num_classes = len(train_ds.classes)
    print('Classes:', train_ds.classes)
    print('Num classes:', num_classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # model
    model = build_model(args.model, num_classes=num_classes, pretrained=True)
    model.to(device)

    # freeze backbone (all params), then unfreeze classifier head
    set_requires_grad(model, False)
    # enable head params
    head_params = get_head_params(model)
    for p in head_params:
        p.requires_grad = True

    # optimizer for head only
    opt = torch.optim.Adam(head_params, lr=args.lr, weight_decay=args.wd)
    
    # Compute class weights to handle class imbalance
    class_weights = compute_class_weights(train_ds).to(device)
    print('Class weights:', {cls: f'{w:.2f}' for cls, w in zip(train_ds.classes, class_weights.tolist())})
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val = 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # training loop
    for epoch in range(1, args.epochs + 1):
        is_finetune = epoch > args.freeze_epochs
        if is_finetune and opt.param_groups[0]['lr'] == args.lr:
            # switch to fine-tune mode: unfreeze all and recreate optimizer with smaller lr
            print('Unfreezing backbone and switching to fine-tune optimizer')
            set_requires_grad(model, True)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=args.wd)

        model.train()
        running_loss = 0.0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        print(f'Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  val_acc={val_acc:.4f}  time={(time.time()-t0):.1f}s')

        # save best
        if val_acc > best_val:
            best_val = val_acc
            ckpt = out_dir / f'best_1{args.model}.pth'
            torch.save({'model_state_dict': model.state_dict(), 'args': vars(args), 'classes': train_ds.classes}, ckpt)

    # final evaluation on test set
    test_ds = WasteDataset(args.data_dir, split='test', transform=get_transforms('torchvision', 'val', args.img_size))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_acc = evaluate(model, test_loader, device)
    print('Best val acc:', best_val)
    print('Test acc:', test_acc)


if __name__ == '__main__':
    train()
