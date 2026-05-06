import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.replay_pad_clip_dataset import ReplayPADClipDataset
from src.models.temporal_lightweight_lstm import TemporalBackboneLSTM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = REPO_ROOT / "outputs" / "predictions" / "upgrade"
RESULT_DIR = REPO_ROOT / "outputs" / "results" / "upgrade"


MODEL_NAME_MAP = {
    "mobilenetv3_small": "CNN-LSTM",
    "minifasnet": "MiniFASNet-LSTM",
    "mobilenetv4_small": "MobileNetV4-LSTM",
    "efficientnet_lite": "EfficientNet-Lite-LSTM",
    "shufflenetv2": "ShuffleNetV2-LSTM",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clip_csv", type=str, default="/home/saslab01/Desktop/replay_pad/frame_index/replayattack_5frame_index.csv")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--input_type", type=str, default="5-frame clip")
    return parser.parse_args()


def build_loader(csv_path, split, img_size, batch_size, num_workers):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = ReplayPADClipDataset(csv_path=csv_path, split=split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def aggregate_clip_to_video(clip_df, out_csv=None):
    agg_dict = {
        "label": "first",
        "split": "first",
        "attack_type": "first",
        "support_type": "first",
        "environment": "first",
        "dataset_name": "first",
        "score": "mean",
    }
    if "client_id" in clip_df.columns:
        agg_dict["client_id"] = "first"

    video_df = clip_df.groupby("video_id", as_index=False).agg(agg_dict)
    if out_csv is not None:
        video_df.to_csv(out_csv, index=False)
    return video_df


def compute_metrics_from_counts(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    apcer = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    acer = (apcer + bpcer) / 2.0
    return {
        "accuracy": accuracy,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def apply_threshold_and_compute_metrics(video_df, threshold):
    pred = (video_df["score"] >= threshold).astype(int)
    label = video_df["label"].astype(int)

    tp = int(((pred == 1) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    return compute_metrics_from_counts(tp, tn, fp, fn)


def search_best_threshold(video_df, step=0.001):
    best_threshold = 0.0
    best_metrics = None
    best_acer = float("inf")

    threshold = 0.0
    while threshold <= 1.0:
        metrics = apply_threshold_and_compute_metrics(video_df, threshold)
        if metrics["acer"] < best_acer:
            best_acer = metrics["acer"]
            best_threshold = threshold
            best_metrics = metrics
        threshold += step

    return best_threshold, best_metrics


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def sanitize_metrics(metrics):
    return {k: v for k, v in metrics.items() if k != "hter"}


def build_threshold_payload(args, threshold, devel_metrics):
    return {
        "model": MODEL_NAME_MAP[args.backbone.lower()],
        "backbone": args.backbone,
        "input_type": args.input_type,
        "threshold_selected_on": "devel",
        "threshold": round(float(threshold), 6),
        "devel_metrics": sanitize_metrics(devel_metrics),
    }


def build_result_payload(args, threshold, devel_metrics, test_metrics, artifacts):
    return {
        "model": MODEL_NAME_MAP[args.backbone.lower()],
        "backbone": args.backbone,
        "input_type": args.input_type,
        "initialization": "random",
        "threshold_selected_on": "devel",
        "threshold": round(float(threshold), 6),
        "devel_threshold_search_result": sanitize_metrics(devel_metrics),
        "test_video_metrics": sanitize_metrics(test_metrics),
        "artifacts": artifacts,
    }


def inference_and_save_clip_predictions(model, csv_path, split, img_size, batch_size, num_workers, tag):
    _, loader = build_loader(csv_path, split, img_size, batch_size, num_workers)
    split_df = pd.read_csv(csv_path)
    split_df = split_df[split_df["split"] == split].reset_index(drop=True)

    all_scores = []
    model.eval()
    with torch.no_grad():
        for clips, _ in loader:
            clips = clips.to(DEVICE, non_blocking=True)
            logits = model(clips)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_scores.extend(probs.detach().cpu().numpy().tolist())

    if len(all_scores) != len(split_df):
        raise ValueError(
            f"Number of scores ({len(all_scores)}) does not match rows ({len(split_df)}) for split={split}"
        )

    split_df["score"] = all_scores
    out_csv = PRED_DIR / f"{tag}_{split}_clip_predictions.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(out_csv, index=False)
    return split_df, str(out_csv)


def main():
    args = parse_args()
    backbone = args.backbone.lower()
    tag = f"{backbone}_lstm_clip5"

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    model = TemporalBackboneLSTM(
        backbone_name=backbone,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        pretrained=False,
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    devel_clip_df, devel_clip_csv = inference_and_save_clip_predictions(
        model, args.clip_csv, "devel", args.img_size, args.batch_size, args.num_workers, tag
    )
    test_clip_df, test_clip_csv = inference_and_save_clip_predictions(
        model, args.clip_csv, "test", args.img_size, args.batch_size, args.num_workers, tag
    )

    devel_video_csv = PRED_DIR / f"{tag}_devel_video_predictions.csv"
    test_video_csv = PRED_DIR / f"{tag}_test_video_predictions.csv"

    devel_video_df = aggregate_clip_to_video(devel_clip_df, devel_video_csv)
    test_video_df = aggregate_clip_to_video(test_clip_df, test_video_csv)

    best_threshold, devel_metrics = search_best_threshold(devel_video_df, step=0.001)
    test_metrics = apply_threshold_and_compute_metrics(test_video_df, best_threshold)

    threshold_json = RESULT_DIR / f"{tag}" / "devel_threshold.json"
    save_json(threshold_json, build_threshold_payload(args, best_threshold, devel_metrics))

    result = build_result_payload(
        args=args,
        threshold=best_threshold,
        devel_metrics=devel_metrics,
        test_metrics=test_metrics,
        artifacts={
            "devel_clip_predictions_csv": devel_clip_csv,
            "test_clip_predictions_csv": test_clip_csv,
            "devel_video_predictions_csv": str(devel_video_csv),
            "test_video_predictions_csv": str(test_video_csv),
            "devel_threshold_json": str(threshold_json),
        },
    )

    save_json(Path(args.output_json), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

