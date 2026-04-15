import argparse
import csv
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.replay_pad_clip_dataset import ReplayPADClipDataset
from src.evaluation.video_level_metrics import (
    apply_threshold_and_compute_metrics,
    search_best_threshold,
)
from src.models.cnn_lstm_baseline import CNNLSTMBinaryClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO_ROOT / "outputs" / "results"
DEVEL_CSV = REPO_ROOT / "clip_index" / "replayattack_clip10_index.csv"
TEST_CSV = REPO_ROOT / "clip_index" / "replayattack_clip10_index.csv"
CHECKPOINT_PATH = REPO_ROOT / "outputs" / "checkpoints" / "cnn_lstm_clip10_random_best.pth"


def rounded(value, digits=6):
    if value is None:
        return None
    return round(float(value), digits)


def build_comparison_row(mode, metrics, threshold, latency_ms, peak_gpu_memory_mb):
    return {
        "Mode": mode,
        "Threshold": rounded(threshold),
        "Accuracy": rounded(metrics["accuracy"]),
        "APCER": rounded(metrics["apcer"]),
        "BPCER": rounded(metrics["bpcer"]),
        "ACER": rounded(metrics["acer"]),
        "Inference_Latency_ms": rounded(latency_ms),
        "Peak_GPU_Memory_MB": (
            rounded(peak_gpu_memory_mb)
            if peak_gpu_memory_mb is not None
            else "N/A"
        ),
    }


def compute_delta_row(baseline, streaming):
    memory_delta = "N/A"
    if baseline["Peak_GPU_Memory_MB"] != "N/A" and streaming["Peak_GPU_Memory_MB"] != "N/A":
        memory_delta = rounded(
            baseline["Peak_GPU_Memory_MB"] - streaming["Peak_GPU_Memory_MB"]
        )

    return {
        "Baseline_Mode": baseline["Mode"],
        "Streaming_Mode": streaming["Mode"],
        "Memory_Reduction_MB": memory_delta,
        "Latency_Increase_ms": rounded(
            streaming["Inference_Latency_ms"] - baseline["Inference_Latency_ms"]
        ),
        "ACER_Delta": rounded(streaming["ACER"] - baseline["ACER"]),
    }


def aggregate_clip_to_video(clip_df):
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
    return clip_df.groupby("video_id", as_index=False).agg(agg_dict)


def build_loader(csv_path, split, img_size, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = ReplayPADClipDataset(
        csv_path=str(csv_path),
        split=split,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_split_df(csv_path, split):
    df = pd.read_csv(csv_path)
    return df[df["split"] == split].reset_index(drop=True)


def predict_split(model, csv_path, split, mode, args, device):
    loader = build_loader(
        csv_path=csv_path,
        split=split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    split_df = load_split_df(csv_path, split)
    scores = []

    model.eval()
    with torch.no_grad():
        for clips, _ in loader:
            clips = clips.to(device, non_blocking=True)
            if mode == "baseline":
                logits = model(clips)
            elif mode == "streaming":
                logits = model.forward_streaming(clips, chunk_size=args.chunk_size)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            probs = torch.softmax(logits, dim=1)[:, 1]
            scores.extend(probs.detach().cpu().numpy().tolist())

    if len(scores) != len(split_df):
        raise ValueError(
            f"Score count mismatch for mode={mode}, split={split}: {len(scores)} vs {len(split_df)}"
        )
    split_df["score"] = scores
    return split_df


def evaluate_mode(model, mode, args, device):
    devel_clip_df = predict_split(model, DEVEL_CSV, "devel", mode, args, device)
    test_clip_df = predict_split(model, TEST_CSV, "test", mode, args, device)

    devel_video_df = aggregate_clip_to_video(devel_clip_df)
    test_video_df = aggregate_clip_to_video(test_clip_df)

    threshold, _ = search_best_threshold(devel_video_df, step=0.001)
    test_metrics = apply_threshold_and_compute_metrics(test_video_df, threshold)
    return threshold, test_metrics


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_latency_ms(model, mode, args, device):
    clip = torch.randn(1, 10, 3, args.img_size, args.img_size, device=device)

    def forward():
        if mode == "baseline":
            return model(clip)
        return model.forward_streaming(clip, chunk_size=args.chunk_size)

    with torch.no_grad():
        for _ in range(args.warmup):
            forward()
        synchronize_if_cuda(device)
        start = time.perf_counter()
        for _ in range(args.iterations):
            forward()
        synchronize_if_cuda(device)
        end = time.perf_counter()
    return (end - start) * 1000.0 / args.iterations


def measure_peak_gpu_memory_mb(model, mode, args, device):
    if device.type != "cuda":
        return None
    clip = torch.randn(1, 10, 3, args.img_size, args.img_size, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        if mode == "baseline":
            model(clip)
        else:
            model.forward_streaming(clip, chunk_size=args.chunk_size)
    synchronize_if_cuda(device)
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def write_summary(path, rows, delta, device, chunk_size):
    content = f"""# CNN-LSTM Memory Reduction Summary

Device: `{device}`

Streaming chunk size: `{chunk_size}`

## Comparison

{markdown_table(rows)}

## Delta

{markdown_table([delta])}

Interpretation: streaming inference keeps the same CNN-LSTM checkpoint and classification path, but extracts MobileNetV3-Small frame features in smaller chunks before the LSTM. The goal is to reduce peak GPU memory while preserving test video-level ACER; latency is recorded as the expected trade-off.
"""
    path.write_text(content)


def load_model(device, args):
    model = CNNLSTMBinaryClassifier(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, args)

    rows = []
    for mode in ["baseline", "streaming"]:
        threshold, metrics = evaluate_mode(model, mode, args, device)
        latency_ms = measure_latency_ms(model, mode, args, device)
        peak_memory_mb = measure_peak_gpu_memory_mb(model, mode, args, device)
        rows.append(
            build_comparison_row(
                mode=mode,
                metrics=metrics,
                threshold=threshold,
                latency_ms=latency_ms,
                peak_gpu_memory_mb=peak_memory_mb,
            )
        )

    delta = compute_delta_row(rows[0], rows[1])
    write_csv(RESULT_DIR / "cnn_lstm_memory_reduction_comparison.csv", rows)
    write_csv(RESULT_DIR / "cnn_lstm_memory_reduction_delta.csv", [delta])
    write_summary(
        RESULT_DIR / "cnn_lstm_memory_reduction_summary.md",
        rows,
        delta,
        device,
        args.chunk_size,
    )


if __name__ == "__main__":
    main()
