import argparse
import json
import math
import shutil
import time
from pathlib import Path

import pandas as pd
import torch

from src.models.temporal_lightweight_lstm import TemporalBackboneLSTM


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_RESULT_JSON = REPO_ROOT / "outputs" / "results" / "cnn_lstm_clip5_eval_results.json"
BASELINE_CHECKPOINT = REPO_ROOT / "outputs" / "checkpoints" / "cnn_lstm_clip5_random_best.pth"
BASELINE_FALLBACK_CHECKPOINT = REPO_ROOT.parent / "replay_pad" / "outputs" / "checkpoints" / "cnn_lstm_clip5_random_best.pth"

MODEL_NAME_MAP = {
    "mobilenetv3_small": "CNN-LSTM",
    "minifasnet": "MiniFASNet-LSTM",
    "mobilenetv4_small": "MobileNetV4-LSTM",
    "efficientnet_lite": "EfficientNet-Lite-LSTM",
    "shufflenetv2": "ShuffleNetV2-LSTM",
}

BACKBONE_DISPLAY = {
    "mobilenetv3_small": "MobileNetV3-Small",
    "minifasnet": "MiniFASNet",
    "mobilenetv4_small": "MobileNetV4",
    "efficientnet_lite": "EfficientNet-Lite",
    "shufflenetv2": "ShuffleNetV2",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    parser.add_argument("--summary_md", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    return parser.parse_args()


def rounded(value, digits=6):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return value
    return round(float(value), digits)


def sanitize_metrics(metrics):
    return {k: v for k, v in metrics.items() if k != "hter"}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_baseline_checkpoint():
    if BASELINE_CHECKPOINT.exists():
        return BASELINE_CHECKPOINT
    if BASELINE_FALLBACK_CHECKPOINT.exists():
        BASELINE_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASELINE_FALLBACK_CHECKPOINT, BASELINE_CHECKPOINT)
        return BASELINE_CHECKPOINT
    raise FileNotFoundError(
        "Could not find baseline checkpoint in the current repo or the adjacent replay_pad repo."
    )


def build_model(backbone, device, hidden_dim, num_layers):
    model = TemporalBackboneLSTM(
        backbone_name=backbone,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=2,
        pretrained=False,
    ).to(device)
    return model


def load_model(backbone, checkpoint_path, device, hidden_dim, num_layers):
    model = build_model(backbone, device, hidden_dim, num_layers)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def measure_latency_ms(model, device, img_size, warmup, iterations):
    clip = torch.randn(1, 5, 3, img_size, img_size, device=device)

    def forward():
        return model(clip)

    with torch.inference_mode():
        for _ in range(warmup):
            forward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            forward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) * 1000.0 / iterations


def measure_peak_gpu_memory_mb(model, device, img_size):
    if device.type != "cuda":
        return None
    clip = torch.randn(1, 5, 3, img_size, img_size, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(clip)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def compute_params(model):
    return sum(p.numel() for p in model.parameters())


def compute_size_mb(params):
    return params * 4 / 1024 / 1024


def build_row(model_name, backbone, input_type, result_json, params, latency_ms, peak_mem_mb):
    metrics = sanitize_metrics(result_json["test_video_metrics"])
    return {
        "Model": model_name,
        "Backbone": backbone,
        "Input type": input_type,
        "Temporal length": 5,
        "Order used": "Yes",
        "Init": "Random",
        "Accuracy": rounded(metrics["accuracy"]),
        "APCER": rounded(metrics["apcer"]),
        "BPCER": rounded(metrics["bpcer"]),
        "ACER": rounded(metrics["acer"]),
        "Params": int(params),
        "Size(MB)": rounded(compute_size_mb(params)),
        "Latency(ms)": rounded(latency_ms),
        "Peak GPU Memory(MB)": rounded(peak_mem_mb) if peak_mem_mb is not None else "N/A",
    }


def markdown_table(rows):
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def compute_tradeoff_rank(rows):
    frame = pd.DataFrame(rows).copy()
    numeric_cols = ["ACER", "Latency(ms)", "Peak GPU Memory(MB)"]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    usable = frame[frame["Peak GPU Memory(MB)"].notna()].copy()
    if usable.empty:
        usable = frame.copy()

    def normalize(series):
        min_v = series.min()
        max_v = series.max()
        if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - min_v) / (max_v - min_v)

    usable["tradeoff_score"] = (
        normalize(usable["ACER"])
        + normalize(usable["Latency(ms)"])
        + normalize(usable["Peak GPU Memory(MB)"].fillna(usable["Peak GPU Memory(MB)"].max()))
    ) / 3.0
    return usable.sort_values("tradeoff_score", ascending=True).iloc[0]["Model"]


def summary_text(rows, skipped):
    frame = pd.DataFrame(rows).copy()
    frame["Accuracy"] = pd.to_numeric(frame["Accuracy"], errors="coerce")
    frame["ACER"] = pd.to_numeric(frame["ACER"], errors="coerce")
    frame["Latency(ms)"] = pd.to_numeric(frame["Latency(ms)"], errors="coerce")
    frame["Peak GPU Memory(MB)"] = pd.to_numeric(frame["Peak GPU Memory(MB)"], errors="coerce")

    best_accuracy = frame.loc[frame["Accuracy"].idxmax(), "Model"]
    best_acer = frame.loc[frame["ACER"].idxmin(), "Model"]
    fastest = frame.loc[frame["Latency(ms)"].idxmin(), "Model"]
    mem_numeric = frame["Peak GPU Memory(MB)"]
    if mem_numeric.notna().any():
        lowest_mem = frame.loc[mem_numeric.idxmin(), "Model"]
    else:
        lowest_mem = "N/A"
    tradeoff = compute_tradeoff_rank(rows)

    lines = [
        "# Temporal Lightweight Backbone Summary",
        "",
        "## 1. Experiment Purpose",
        "Same 5-frame temporal-order condition, backbone only changes, to compare the performance-efficiency trade-off.",
        "",
        "## 2. Experimental Conditions",
        "- Replay-Attack only",
        "- 5-frame clip input",
        "- temporal order preserved through an LSTM head",
        "- random initialization only",
        "- threshold chosen on devel video-level scores only",
        "- test evaluated at video level with the fixed devel threshold",
        "- HTER excluded completely",
        "",
        "## 3. Performance Comparison",
        markdown_table(rows),
        "",
        "Accuracy, APCER, BPCER, and ACER are reported at the video level. ACER is the primary detection metric for interpretation.",
        "",
        "## 4. Efficiency Comparison",
        "Latency(ms) is the forward-pass time for one `[1, 5, 3, 224, 224]` clip. Peak GPU Memory(MB) is the peak forward-pass allocation when CUDA is available.",
        "",
        "## 5. Conclusions",
        f"- Most accurate model: {best_accuracy}",
        f"- Lowest ACER: {best_acer}",
        f"- Fastest model: {fastest}",
        f"- Lowest memory model: {lowest_mem}",
        f"- Best trade-off model: {tradeoff}",
        "",
        "## 6. Baseline Comparison",
        "The first row is the existing 5-frame CNN-LSTM baseline using MobileNetV3-Small.",
    ]

    if skipped:
        lines += ["", "## 7. Skipped Models"] + [f"- {item}" for item in skipped]

    lines += [
        "",
        "## 8. Commands",
        "```bash",
        "python -m src.engine.train_temporal_lightweight_lstm \\",
        "  --backbone minifasnet \\",
        "  --clip_csv frame_index/replayattack_5frame_index.csv \\",
        "  --epochs 10 \\",
        "  --batch_size 8 \\",
        "  --lr 0.0001 \\",
        "  --img_size 224 \\",
        "  --seed 42 \\",
        "  --save_path outputs/checkpoints/upgrade/minifasnet_lstm_clip5_random_best.pth",
        "",
        "python -m src.engine.evaluate_temporal_lightweight_lstm \\",
        "  --backbone minifasnet \\",
        "  --checkpoint outputs/checkpoints/upgrade/minifasnet_lstm_clip5_random_best.pth \\",
        "  --clip_csv frame_index/replayattack_5frame_index.csv \\",
        "  --output_json outputs/results/upgrade/minifasnet_lstm_clip5_eval_results.json",
        "",
        "python -m src.analysis.compare_temporal_lightweight_efficiency \\",
        "  --results_dir outputs/results/upgrade \\",
        "  --checkpoints_dir outputs/checkpoints/upgrade \\",
        "  --output_csv outputs/results/upgrade/temporal_lightweight_backbone_comparison.csv \\",
        "  --output_md outputs/results/upgrade/temporal_lightweight_backbone_comparison.md \\",
        "  --summary_md outputs/results/upgrade/temporal_lightweight_backbone_summary.md",
        "```",
    ]
    return "\n".join(lines)


def gather_rows(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    checkpoints_dir = Path(args.checkpoints_dir)

    rows = []
    skipped = []

    baseline_checkpoint = resolve_baseline_checkpoint()
    baseline_result = load_json(BASELINE_RESULT_JSON)
    baseline_model = load_model("mobilenetv3_small", baseline_checkpoint, device, args.hidden_dim, args.num_layers)
    baseline_params = compute_params(baseline_model)
    baseline_latency = measure_latency_ms(baseline_model, device, args.img_size, args.warmup, args.iterations)
    baseline_mem = measure_peak_gpu_memory_mb(baseline_model, device, args.img_size)
    rows.append(
        build_row(
            "CNN-LSTM",
            BACKBONE_DISPLAY["mobilenetv3_small"],
            baseline_result.get("input_type", "5-frame clip"),
            baseline_result,
            baseline_params,
            baseline_latency,
            baseline_mem,
        )
    )

    for backbone in ["minifasnet", "mobilenetv4_small", "efficientnet_lite", "shufflenetv2"]:
        result_path = results_dir / f"{backbone}_lstm_clip5_eval_results.json"
        checkpoint_path = checkpoints_dir / f"{backbone}_lstm_clip5_random_best.pth"
        if not result_path.exists():
            skipped.append(
                f"{MODEL_NAME_MAP[backbone]} skipped: missing result JSON at {result_path}"
            )
            continue
        if not checkpoint_path.exists():
            skipped.append(
                f"{MODEL_NAME_MAP[backbone]} skipped: missing checkpoint at {checkpoint_path}"
            )
            continue

        result_json = load_json(result_path)
        model = load_model(backbone, checkpoint_path, device, args.hidden_dim, args.num_layers)
        params = compute_params(model)
        latency = measure_latency_ms(model, device, args.img_size, args.warmup, args.iterations)
        peak_mem = measure_peak_gpu_memory_mb(model, device, args.img_size)
        rows.append(
            build_row(
                MODEL_NAME_MAP[backbone],
                BACKBONE_DISPLAY[backbone],
                result_json.get("input_type", "5-frame clip"),
                result_json,
                params,
                latency,
                peak_mem,
            )
        )

    return rows, skipped


def main():
    args = parse_args()
    rows, skipped = gather_rows(args)

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    summary_md = Path(args.summary_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)

    write_csv(output_csv, rows)
    output_md.write_text(markdown_table(rows) + "\n")
    summary_md.write_text(summary_text(rows, skipped) + "\n")

    print(markdown_table(rows))
    if skipped:
        print("\nSkipped models:")
        for item in skipped:
            print(f"- {item}")


if __name__ == "__main__":
    main()
