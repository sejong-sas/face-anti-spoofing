from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

ROOT = Path("/home/saslab01/Desktop/replay_pad")
CSV_PATH = ROOT / "metadata" / "frames_10frame.csv"
CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "image_baseline_1frame_resnet18.pth"

PRED_CSV_PATH = ROOT / "outputs" / "predictions" / "image_baseline_10frame_test_predictions.csv"
VIDEO_PRED_CSV_PATH = ROOT / "outputs" / "predictions" / "image_baseline_10frame_test_video_predictions.csv"
METRIC_CSV_PATH = ROOT / "outputs" / "metrics" / "image_baseline_10frame_test_metrics.csv"
CM_FIG_PATH = ROOT / "outputs" / "figures" / "image_baseline_10frame_confusion_matrix.png"

IMG_SIZE = 224
BATCH_SIZE = 32


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
        video_id = row["video_id"]
        attack_type = row["attack_type"]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, video_id, attack_type


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_model(device):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def compute_metrics(y_true, y_pred, y_score):
    acc = (y_true == y_pred).mean()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    apcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acer = (apcer + bpcer) / 2.0

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = 0.0

    return {
        "accuracy": acc,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }, cm


def save_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["real", "attack"])
    ax.set_yticklabels(["real", "attack"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    df = pd.read_csv(CSV_PATH)
    test_df = df[df["split"] == "test"].copy()
    print(f"[INFO] test frames: {len(test_df)}")

    dataset = FrameDataset(test_df, transform=get_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = build_model(device)
    softmax = nn.Softmax(dim=1)

    rows = []

    with torch.no_grad():
        for images, labels, video_ids, attack_types in loader:
            images = images.to(device)
            logits = model(images)
            probs = softmax(logits)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            labels = labels.numpy()

            for i in range(len(video_ids)):
                rows.append({
                    "sample_id": f"{attack_types[i]}/{video_ids[i]}",
                    "video_id": video_ids[i],
                    "attack_type": attack_types[i],
                    "label": int(labels[i]),
                    "score_attack": float(probs[i]),
                    "pred": int(preds[i]),
                })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(PRED_CSV_PATH, index=False)
    print(f"[INFO] saved frame-level predictions -> {PRED_CSV_PATH}")

    video_df = pred_df.groupby("sample_id").agg({
        "video_id": "first",
        "attack_type": "first",
        "label": "first",
        "score_attack": "mean"
    }).reset_index()

    video_df["pred"] = (video_df["score_attack"] >= 0.5).astype(int)
    video_df.to_csv(VIDEO_PRED_CSV_PATH, index=False)
    print(f"[INFO] saved video-level predictions -> {VIDEO_PRED_CSV_PATH}")
    print(f"[INFO] unique test videos after grouping: {len(video_df)}")

    y_true = video_df["label"].values
    y_pred = video_df["pred"].values
    y_score = video_df["score_attack"].values

    metrics, cm = compute_metrics(y_true, y_pred, y_score)
    pd.DataFrame([metrics]).to_csv(METRIC_CSV_PATH, index=False)
    print(f"[INFO] saved metrics -> {METRIC_CSV_PATH}")

    save_confusion_matrix(cm, CM_FIG_PATH)
    print(f"[INFO] saved confusion matrix -> {CM_FIG_PATH}")

    print("\n===== Final Test Metrics (Video-level) =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()