"""Train the currency classifier on a laptop GPU.

Expects:
    data/currency/train/<class>/*.jpg
    data/currency/val/<class>/*.jpg
(collect_currency.py + split_dataset.py create this layout.)

Example:
    python train_currency.py --arch mobilenet_v3_large --epochs 15
"""
import argparse
import os
import time

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from currency import ARCHS, build_model, eval_transform, train_transform
from device import pick_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/currency")
    p.add_argument("--arch", default="mobilenet_v3_large", choices=ARCHS)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=16, help="lower this if you run out of GPU memory")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="models/currency.pt")
    return p.parse_args()


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    use_amp = device == "cuda"
    total_loss, correct, seen = 0.0, 0, 0
    preds_all, labels_all = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training), \
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if training:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * y.size(0)
        correct += (preds == y).sum().item()
        seen += y.size(0)
        preds_all += preds.tolist()
        labels_all += y.tolist()
    return total_loss / seen, correct / seen, preds_all, labels_all


def main():
    args = parse_args()
    device = pick_device()
    print(f"device: {device}")

    train_ds = ImageFolder(os.path.join(args.data, "train"), train_transform())
    val_ds = ImageFolder(os.path.join(args.data, "val"), eval_transform())
    assert train_ds.classes == val_ds.classes, "train and val must have the same class folders"
    classes = train_ds.classes
    print(f"classes: {classes}  train={len(train_ds)}  val={len(val_ds)}")

    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_dl = DataLoader(val_ds, args.batch, num_workers=args.workers, pin_memory=True)

    model = build_model(args.arch, len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(model, train_dl, device, criterion, optimizer, scaler)
        va_loss, va_acc, _, _ = run_epoch(model, val_dl, device, criterion)
        scheduler.step()
        print(f"epoch {epoch:2d}/{args.epochs}  train loss {tr_loss:.3f} acc {tr_acc:.3f}  "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}  ({time.time() - start:.0f}s)")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"arch": args.arch, "classes": classes, "state_dict": model.state_dict()}, args.out)
            print(f"  saved best model -> {args.out}")

    # Final report on the best checkpoint: these numbers go straight into the lab report.
    model.load_state_dict(torch.load(args.out, map_location=device)["state_dict"])
    _, acc, preds, labels = run_epoch(model, val_dl, device, criterion)
    print(f"\nbest val accuracy: {acc:.4f}\n")
    print(classification_report(labels, preds, target_names=classes, digits=3))
    print("confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
