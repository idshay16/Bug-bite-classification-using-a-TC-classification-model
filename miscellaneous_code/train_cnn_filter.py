#!/usr/bin/env python3
"""
EfficientNet-B0 binary classifier: organized (positive) vs disorganized (negative).

Input dirs:
  cnn_labels/positive/   -- organized cyclone images
  cnn_labels/negative/   -- disorganized / artifact images

Usage:
  python miscellaneous_code/train_cnn_filter.py
  python miscellaneous_code/train_cnn_filter.py --data cnn_labels --epochs 20 --batch 32
  python miscellaneous_code/train_cnn_filter.py --resume cnn_filter_model.pth
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
IMAGE_SIZE = 224


class BinaryImageDataset(Dataset):
    def __init__(self, pos_dir: Path, neg_dir: Path, transform=None):
        self.samples = []
        self.transform = transform
        for p in pos_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                self.samples.append((p, 1))
        for p in neg_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                self.samples.append((p, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_transforms(augment: bool):
    if augment:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_model(num_classes: int = 2, freeze_backbone: bool = False) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            out = model(imgs)
            loss = criterion(out, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        pred = out.argmax(1)
        correct += (pred == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        pred = out.argmax(1)
        correct += (pred == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    default="cnn_labels",         help="Root dir with positive/ and negative/")
    ap.add_argument("--out",     default="cnn_filter_model.pth", help="Output model path")
    ap.add_argument("--epochs",  type=int,   default=15)
    ap.add_argument("--batch",   type=int,   default=32)
    ap.add_argument("--lr",      type=float, default=1e-4)
    ap.add_argument("--val",     type=float, default=0.15,      help="Validation fraction")
    ap.add_argument("--workers", type=int,   default=4,         help="DataLoader workers")
    ap.add_argument("--freeze",  action="store_true",           help="Freeze backbone, train head only (faster warmup)")
    ap.add_argument("--resume",  default=None,                  help="Resume from checkpoint path")
    ap.add_argument("--no-amp",  action="store_true",           help="Disable mixed precision")
    args = ap.parse_args()

    data_dir = Path(args.data)
    pos_dir  = data_dir / "positive"
    neg_dir  = data_dir / "negative"

    if not pos_dir.exists() or not neg_dir.exists():
        sys.exit(f"Need {pos_dir} and {neg_dir}. Run label_for_cnn.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp
    print(f"Device: {device}  AMP: {use_amp}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    # Dataset
    full_ds = BinaryImageDataset(pos_dir, neg_dir, transform=None)
    n = len(full_ds)
    if n < 10:
        sys.exit(f"Only {n} images found. Label more before training.")

    n_val   = max(1, int(n * args.val))
    n_train = n - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    pos = sum(1 for _, l in full_ds.samples if l == 1)
    neg = n - pos
    print(f"Dataset: {n} total  |  positive={pos}  negative={neg}")
    print(f"Split:   train={n_train}  val={n_val}")

    train_ds.dataset = BinaryImageDataset.__new__(BinaryImageDataset)
    # Rebuild with correct transforms per split
    train_set = BinaryImageDataset(pos_dir, neg_dir, transform=build_transforms(True))
    val_set   = BinaryImageDataset(pos_dir, neg_dir, transform=build_transforms(False))

    # Apply same split indices
    train_set = torch.utils.data.Subset(
        BinaryImageDataset(pos_dir, neg_dir, transform=build_transforms(True)),
        train_ds.indices,
    )
    val_set = torch.utils.data.Subset(
        BinaryImageDataset(pos_dir, neg_dir, transform=build_transforms(False)),
        val_ds.indices,
    )

    loader_kw = dict(batch_size=args.batch, num_workers=args.workers,
                     pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_kw)

    # Model
    model = build_model(freeze_backbone=args.freeze).to(device)
    start_epoch = 0
    best_val_acc = 0.0
    history = []

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        history = ckpt.get("history", [])
        print(f"Resumed from {args.resume}  (epoch {start_epoch}, best_val_acc={best_val_acc:.4f})")

    # Class weights to handle imbalance
    weight = torch.tensor([1.0 / max(neg, 1), 1.0 / max(pos, 1)], device=device)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    out_path = Path(args.out)

    print(f"\nTraining EfficientNet-B0 for {args.epochs} epochs...")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        marker = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "best_val_acc": best_val_acc,
                "history": history,
                "args": vars(args),
            }, out_path)
            marker = " *saved*"

        entry = {"epoch": epoch + 1, "tr_loss": round(tr_loss, 4), "tr_acc": round(tr_acc, 4),
                 "vl_loss": round(vl_loss, 4), "vl_acc": round(vl_acc, 4)}
        history.append(entry)
        print(f"  Epoch {epoch+1:3d}  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}{marker}")

    print(f"\nBest val acc: {best_val_acc:.4f}")
    print(f"Model saved → {out_path.resolve()}")

    hist_path = out_path.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"History    → {hist_path}")


if __name__ == "__main__":
    main()
