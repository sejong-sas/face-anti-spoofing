import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

ROOT = Path("/home/saslab01/Desktop/replay_pad")
CSV_PATH = ROOT / "metadata" / "frames_1frame.csv"
CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "image_baseline_1frame_resnet18.pth"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FrameDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["frame_path"]).convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model = model.to(device)
    return model


def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += images.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    df = pd.read_csv(CSV_PATH)

    train_df = df[df["split"] == "train"].copy()
    devel_df = df[df["split"] == "devel"].copy()

    print(f"[INFO] train face frames: {len(train_df)}")
    print(f"[INFO] devel face frames: {len(devel_df)}")

    train_tf, eval_tf = get_transforms()

    train_dataset = FrameDataset(train_df, transform=train_tf)
    devel_dataset = FrameDataset(devel_df, transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    devel_loader = DataLoader(devel_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_devel_loss = float("inf")
    best_epoch = -1
    best_devel_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")

        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        devel_loss, devel_acc = run_one_epoch(
            model, devel_loader, criterion, optimizer, device, train=False
        )

        print(f"[Epoch {epoch + 1}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        print(f"[Epoch {epoch + 1}] devel_loss={devel_loss:.4f} devel_acc={devel_acc:.4f}")

        if devel_loss < best_devel_loss:
            best_devel_loss = devel_loss
            best_devel_acc = devel_acc
            best_epoch = epoch + 1

            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "devel_loss": best_devel_loss,
                "devel_acc": best_devel_acc,
            }, CHECKPOINT_PATH)

            print(f"[INFO] Best model saved -> {CHECKPOINT_PATH}")

    print("\n[INFO] Training finished")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Devel Loss: {best_devel_loss:.6f}")
    print(f"Best Devel Acc: {best_devel_acc:.6f}")
    print(f"Saved checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()