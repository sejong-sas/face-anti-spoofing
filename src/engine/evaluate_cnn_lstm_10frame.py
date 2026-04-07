from pathlib import Path

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = Path("/home/saslab01/Desktop/replay_pad")
CSV_PATH = ROOT / "metadata" / "clips_10frame.csv"
CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "cnn_lstm_10frame.pth"

PRED_CSV_PATH = ROOT / "outputs" / "predictions" / "cnn_lstm_10frame_test_predictions.csv"
METRIC_CSV_PATH = ROOT / "outputs" / "metrics" / "cnn_lstm_10frame_test_metrics.csv"
CM_FIG_PATH = ROOT / "outputs" / "figures" / "cnn_lstm_10frame_confusion_matrix.png"

SEQ_LEN = 10
IMG_SIZE = 224
BATCH_SIZE = 8
HIDDEN_SIZE = 256


class ClipDataset(Dataset):
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
        video_id = row["video_id"]
        attack_type = row["attack_type"]
        sample_id = f"{attack_type}/{video_id}"
        return frames, label, video_id, attack_type, sample_id


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


class CNNLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=None)
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
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.feature_extractor(x)
        feat = feat.view(B, T, -1)
        out, _ = self.lstm(feat)
        last_out = out[:, -1, :]
        logits = self.classifier(last_out)
        return logits


def build_model(device):
    model = CNNLSTM(hidden_size=HIDDEN_SIZE)
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
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
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
    print(f"[INFO] test clips: {len(test_df)}")

    ds = ClipDataset(test_df, get_transform())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = build_model(device)
    softmax = nn.Softmax(dim=1)

    rows = []
    with torch.no_grad():
        for clips, labels, video_ids, attack_types, sample_ids in loader:
            clips = clips.to(device)
            logits = model(clips)
            probs = softmax(logits)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            labels = labels.numpy()

            for i in range(len(video_ids)):
                rows.append({
                    "sample_id": sample_ids[i],
                    "video_id": video_ids[i],
                    "attack_type": attack_types[i],
                    "label": int(labels[i]),
                    "score_attack": float(probs[i]),
                    "pred": int(preds[i]),
                })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(PRED_CSV_PATH, index=False)
    print(f"[INFO] saved predictions -> {PRED_CSV_PATH}")
    print(f"[INFO] unique test videos after grouping: {len(pred_df)}")

    y_true = pred_df["label"].values
    y_pred = pred_df["pred"].values
    y_score = pred_df["score_attack"].values

    metrics, cm = compute_metrics(y_true, y_pred, y_score)
    pd.DataFrame([metrics]).to_csv(METRIC_CSV_PATH, index=False)
    save_confusion_matrix(cm, CM_FIG_PATH)

    print("\n===== Final Test Metrics (Video-level) =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()