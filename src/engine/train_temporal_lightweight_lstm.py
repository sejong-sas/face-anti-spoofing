import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.replay_pad_clip_dataset import ReplayPADClipDataset
from src.models.temporal_lightweight_lstm import TemporalBackboneLSTM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--clip_csv", type=str, default="/home/saslab01/Desktop/replay_pad/frame_index/replayattack_5frame_index.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loader(csv_path, split, img_size, batch_size, num_workers):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = ReplayPADClipDataset(csv_path=csv_path, split=split, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_one_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for clips, labels in loader:
        clips = clips.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(clips)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * clips.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += clips.size(0)

    return total_loss / total_count, total_correct / total_count


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    train_loader = build_loader(args.clip_csv, "train", args.img_size, args.batch_size, args.num_workers)
    devel_loader = build_loader(args.clip_csv, "devel", args.img_size, args.batch_size, args.num_workers)

    model = TemporalBackboneLSTM(
        backbone_name=args.backbone,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        pretrained=False,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_devel_acc = -1.0

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Backbone: {args.backbone}")
    print(f"[INFO] Train CSV: {args.clip_csv}")
    print(f"[INFO] Train clips: {len(train_loader.dataset)}")
    print(f"[INFO] Devel clips: {len(devel_loader.dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer)
        devel_loss, devel_acc = run_one_epoch(model, devel_loader, criterion)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"devel_loss={devel_loss:.4f} devel_acc={devel_acc:.4f}"
        )

        if devel_acc > best_devel_acc:
            best_devel_acc = devel_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"[INFO] Saved best checkpoint -> {args.save_path}")

    print(f"[INFO] Training finished. Best devel acc = {best_devel_acc:.4f}")


if __name__ == "__main__":
    main()

