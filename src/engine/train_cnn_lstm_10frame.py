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
CSV_PATH = ROOT / "metadata" / "clips_10frame.csv"
CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "cnn_lstm_10frame.pth"

BATCH_SIZE = 8
NUM_EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224
SEED = 42
SEQ_LEN = 10
HIDDEN_SIZE = 256


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames = []

        for i in range(SEQ_LEN):
            img = Image.open(row[f"frame_{i}"]).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames, dim=0)
        label = int(row["label"])
        return frames, label


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


class CNNLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_classes=2):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feat = self.feature_extractor(x)     # [B*T, 512, 1, 1]
        feat = feat.view(B, T, -1)           # [B, T, 512]

        out, _ = self.lstm(feat)             # [B, T, hidden]
        last_out = out[:, -1, :]             # [B, hidden]
        logits = self.classifier(last_out)   # [B, 2]
        return logits


def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(clips)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * clips.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += clips.size(0)

    return total_loss / total_count, total_correct / total_count


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    df = pd.read_csv(CSV_PATH)
    train_df = df[df["split"] == "train"].copy()
    devel_df = df[df["split"] == "devel"].copy()

    print(f"[INFO] train clips: {len(train_df)}")
    print(f"[INFO] devel clips: {len(devel_df)}")

    train_tf, eval_tf = get_transforms()
    train_ds = ClipDataset(train_df, train_tf)
    devel_ds = ClipDataset(devel_df, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    devel_loader = DataLoader(devel_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = CNNLSTM(hidden_size=HIDDEN_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_devel_loss = float("inf")
    best_epoch = -1
    best_devel_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")

        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        devel_loss, devel_acc = run_one_epoch(model, devel_loader, criterion, optimizer, device, train=False)

        print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        print(f"[Epoch {epoch+1}] devel_loss={devel_loss:.4f} devel_acc={devel_acc:.4f}")

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


if __name__ == "__main__":
    main()