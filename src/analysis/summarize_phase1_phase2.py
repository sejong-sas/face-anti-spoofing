import argparse
import csv
import json
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO_ROOT / "outputs" / "results"

MODEL_CONFIGS = [
    {
        "key": "1frame",
        "name": "1-frame",
        "input_type": "single frame",
        "temporal_length": 1,
        "order_used": "No",
        "result_json": RESULT_DIR / "mobilenetv3_small_1frame_random_eval_results.json",
        "model_kind": "mobilenet",
    },
    {
        "key": "5frame_avg",
        "name": "5-frame avg",
        "input_type": "5-frame average",
        "temporal_length": 5,
        "order_used": "No",
        "result_json": RESULT_DIR / "mobilenetv3_small_5frame_avg_eval_results.json",
        "model_kind": "mobilenet_avg",
    },
    {
        "key": "10frame_avg",
        "name": "10-frame avg",
        "input_type": "10-frame average",
        "temporal_length": 10,
        "order_used": "No",
        "result_json": RESULT_DIR / "mobilenetv3_small_10frame_avg_eval_results.json",
        "model_kind": "mobilenet_avg",
    },
    {
        "key": "10frame_cnn_lstm",
        "name": "10-frame CNN-LSTM",
        "input_type": "10-frame clip",
        "temporal_length": 10,
        "order_used": "Yes",
        "result_json": RESULT_DIR / "cnn_lstm_clip10_eval_results.json",
        "model_kind": "cnn_lstm",
    },
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def rounded(value, digits=6):
    if value is None:
        return None
    return round(float(value), digits)


def build_performance_row(
    name,
    result,
    input_type,
    temporal_length,
    order_used,
    threshold_source,
):
    metrics = result["test_video_metrics"]
    return {
        "Model": name,
        "Input type": input_type,
        "Temporal length": temporal_length,
        "Order used": order_used,
        "Initialization": result.get("initialization", "unknown"),
        "Threshold source": threshold_source,
        "Threshold": rounded(result["threshold"]),
        "Accuracy": rounded(metrics["accuracy"]),
        "APCER": rounded(metrics["apcer"]),
        "BPCER": rounded(metrics["bpcer"]),
        "ACER": rounded(metrics["acer"]),
    }


def fp32_model_size_mb(params):
    return round((int(params) * 4) / (1024 * 1024), 6)


def build_efficiency_row(
    perf_row,
    params,
    latency_ms,
    per_frame_latency_ms,
    peak_gpu_memory_mb,
):
    return {
        "Model": perf_row["Model"],
        "Input type": perf_row["Input type"],
        "Temporal length": perf_row["Temporal length"],
        "Params": int(params),
        "FP32_Model_Size_MB": fp32_model_size_mb(params),
        "Inference_Latency_ms": rounded(latency_ms, 6),
        "Per_Frame_Latency_ms": (
            rounded(per_frame_latency_ms, 6)
            if per_frame_latency_ms is not None
            else "N/A"
        ),
        "Latency_Unit": "per video/sequence",
        "Peak_GPU_Memory_MB": (
            rounded(peak_gpu_memory_mb, 6)
            if peak_gpu_memory_mb is not None
            else "N/A"
        ),
        "Threshold": perf_row["Threshold"],
        "Accuracy": perf_row["Accuracy"],
        "APCER": perf_row["APCER"],
        "BPCER": perf_row["BPCER"],
        "ACER": perf_row["ACER"],
    }


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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def cuda_peak_memory_mb():
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def make_models(device):
    from src.models.cnn_lstm_baseline import CNNLSTMBinaryClassifier
    from src.models.mobilenetv3_small_baseline import MobileNetV3SmallBinaryClassifier

    mobilenet = MobileNetV3SmallBinaryClassifier(
        num_classes=2,
        pretrained=False,
    ).to(device)
    cnn_lstm = CNNLSTMBinaryClassifier(
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        pretrained=False,
    ).to(device)
    mobilenet.eval()
    cnn_lstm.eval()
    return mobilenet, cnn_lstm


def load_checkpoint_if_available(model, checkpoint_path, device):
    import torch

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))


def synchronize_if_cuda(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_latency_ms(forward_fn, device, warmup, iterations):
    import torch

    with torch.no_grad():
        for _ in range(warmup):
            forward_fn()
        synchronize_if_cuda(device)
        start = time.perf_counter()
        for _ in range(iterations):
            forward_fn()
        synchronize_if_cuda(device)
        end = time.perf_counter()
    return (end - start) * 1000.0 / iterations


def measure_peak_memory_mb(forward_fn, device):
    import torch

    if device.type != "cuda":
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        forward_fn()
    synchronize_if_cuda(device)
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def measure_efficiency(warmup=10, iterations=50):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobilenet, cnn_lstm = make_models(device)
    load_checkpoint_if_available(
        mobilenet,
        REPO_ROOT / "outputs" / "checkpoints" / "mobilenetv3_small_1frame_random_best.pth",
        device,
    )
    load_checkpoint_if_available(
        cnn_lstm,
        REPO_ROOT / "outputs" / "checkpoints" / "cnn_lstm_clip10_random_best.pth",
        device,
    )

    image = torch.randn(1, 3, 224, 224, device=device)
    clip10 = torch.randn(1, 10, 3, 224, 224, device=device)

    def mobile_forward():
        return mobilenet(image)

    def mobile_repeat_forward(frames):
        def _forward():
            scores = []
            for _ in range(frames):
                scores.append(mobilenet(image))
            return torch.stack(scores).mean(dim=0)

        return _forward

    def lstm_forward():
        return cnn_lstm(clip10)

    one_frame_latency = measure_latency_ms(
        mobile_forward, device, warmup, iterations
    )
    one_frame_peak = measure_peak_memory_mb(mobile_forward, device)

    five_frame_forward = mobile_repeat_forward(5)
    ten_frame_forward = mobile_repeat_forward(10)

    return {
        "device": str(device),
        "per_frame_mobilenet_latency_ms": rounded(one_frame_latency, 6),
        "rows": {
            "1frame": {
                "params": count_params(mobilenet),
                "latency_ms": one_frame_latency,
                "per_frame_latency_ms": one_frame_latency,
                "peak_gpu_memory_mb": one_frame_peak,
            },
            "5frame_avg": {
                "params": count_params(mobilenet),
                "latency_ms": measure_latency_ms(
                    five_frame_forward, device, warmup, iterations
                ),
                "per_frame_latency_ms": one_frame_latency,
                "peak_gpu_memory_mb": measure_peak_memory_mb(
                    five_frame_forward, device
                ),
            },
            "10frame_avg": {
                "params": count_params(mobilenet),
                "latency_ms": measure_latency_ms(
                    ten_frame_forward, device, warmup, iterations
                ),
                "per_frame_latency_ms": one_frame_latency,
                "peak_gpu_memory_mb": measure_peak_memory_mb(
                    ten_frame_forward, device
                ),
            },
            "10frame_cnn_lstm": {
                "params": count_params(cnn_lstm),
                "latency_ms": measure_latency_ms(
                    lstm_forward, device, warmup, iterations
                ),
                "per_frame_latency_ms": None,
                "peak_gpu_memory_mb": measure_peak_memory_mb(lstm_forward, device),
            },
        },
    }


def build_performance_rows():
    rows = []
    loaded = {}
    for cfg in MODEL_CONFIGS:
        result = load_json(cfg["result_json"])
        loaded[cfg["key"]] = result
        threshold_source = result.get("threshold_selected_on", "unknown")
        rows.append(
            build_performance_row(
                name=cfg["name"],
                result=result,
                input_type=cfg["input_type"],
                temporal_length=cfg["temporal_length"],
                order_used=cfg["order_used"],
                threshold_source=threshold_source,
            )
        )
    return rows, loaded


def build_efficiency_rows(performance_rows, efficiency):
    rows = []
    perf_by_name = {row["Model"]: row for row in performance_rows}
    for cfg in MODEL_CONFIGS:
        measured = efficiency["rows"][cfg["key"]]
        rows.append(
            build_efficiency_row(
                perf_row=perf_by_name[cfg["name"]],
                params=measured["params"],
                latency_ms=measured["latency_ms"],
                per_frame_latency_ms=measured["per_frame_latency_ms"],
                peak_gpu_memory_mb=measured["peak_gpu_memory_mb"],
            )
        )
    return rows


def write_rule_memo(path):
    content = """# Devel Threshold and Test Evaluation Rule

- Threshold selection: each evaluation script computes spoof scores on the devel split, aggregates them at video level, and searches thresholds from 0.0 to 1.0 with step 0.001 to minimize devel ACER.
- Test evaluation: the selected devel threshold is fixed and applied to the test video-level scores. No threshold is re-selected on the test split.
- Video-level aggregation: 1-frame predictions are grouped by `video_id` and averaged. The 5-frame and 10-frame average settings first average frame scores within each sequence, then group sequence scores by `video_id` and average again. CNN-LSTM first predicts clip scores, then groups clip scores by `video_id` and averages.
- Code check: the current code matches the intended rule for the four requested models. Result JSON files also record `threshold_selected_on: devel`.
"""
    path.write_text(content)


def write_structure_memo(path):
    content = """# CNN-LSTM Structure Memo

The CNN-LSTM model uses MobileNetV3-Small as a frame-level feature extractor. In the current code, the model instantiates `torchvision.models.mobilenet_v3_small`, keeps only `backbone.features`, and does not use the original MobileNetV3 classifier head.

For each input clip `x` with shape `[B, T, C, H, W]`, the frames are reshaped to `[B*T, C, H, W]` and passed through `backbone.features`. Adaptive average pooling maps the final convolutional feature map to one vector per frame. The resulting frame-level feature dimension is 576.

The temporal module is a one-layer LSTM with `input_size=576`, `hidden_size=128`, and `batch_first=True`. The model uses the LSTM output at the last time step as the clip representation, and a linear classifier maps this 128-dimensional vector to two logits for binary real/spoof classification.

Paper-ready wording: We used MobileNetV3-Small up to its convolutional feature extractor as the per-frame backbone and removed the original classifier head. Each frame was represented by a 576-dimensional pooled feature vector, and the sequence of 10 frame features was passed to a one-layer LSTM with a 128-dimensional hidden state. The final hidden output was classified by a linear layer into real and spoof classes.
"""
    path.write_text(content)


def write_summary_markdown(path, performance_rows, efficiency_rows, efficiency):
    one = next(r for r in efficiency_rows if r["Model"] == "1-frame")
    lstm = next(r for r in efficiency_rows if r["Model"] == "10-frame CNN-LSTM")
    acer_delta = rounded(one["ACER"] - lstm["ACER"], 6)
    latency_delta = rounded(lstm["Inference_Latency_ms"] - one["Inference_Latency_ms"], 6)
    if one["Peak_GPU_Memory_MB"] == "N/A" or lstm["Peak_GPU_Memory_MB"] == "N/A":
        memory_delta = "N/A"
    else:
        memory_delta = rounded(lstm["Peak_GPU_Memory_MB"] - one["Peak_GPU_Memory_MB"], 6)

    content = f"""# Phase 1 and Phase 2 Summary

## Target Files

- Model definitions: `src/models/mobilenetv3_small_baseline.py`, `src/models/cnn_lstm_baseline.py`
- Evaluation scripts: `src/engine/evaluate_image_1frame.py`, `src/engine/evaluate_image_5frame_avg.py`, `src/engine/evaluate_image_10frame_avg.py`, `src/engine/evaluate_cnn_lstm.py`
- Result JSON files: `outputs/results/mobilenetv3_small_1frame_random_eval_results.json`, `outputs/results/mobilenetv3_small_5frame_avg_eval_results.json`, `outputs/results/mobilenetv3_small_10frame_avg_eval_results.json`, `outputs/results/cnn_lstm_clip10_eval_results.json`
- Prediction CSV files: `outputs/predictions/image_1frame_random_*`, `outputs/predictions/image_5frame_avg_*`, `outputs/predictions/image_10frame_avg_*`, `outputs/predictions/cnn_lstm_clip10_*`

## Performance Table

{markdown_table(performance_rows)}

## Efficiency Table

Measured device: `{efficiency["device"]}`

{markdown_table(efficiency_rows)}

## Efficiency Measurement Notes

- Parameter count is the number of trainable and non-trainable parameters in the instantiated PyTorch model.
- FP32 model size is computed as `Params * 4 / 1024 / 1024`.
- Inference latency is measured with dummy tensors at 224x224 resolution after warmup, using `torch.no_grad()`. `Inference_Latency_ms` is the per-video or per-sequence value.
- `Per_Frame_Latency_ms` is reported for the MobileNetV3-Small frame model and its frame-averaging variants. It is `N/A` for CNN-LSTM because the measured unit is the whole 10-frame clip.
- For 5-frame avg and 10-frame avg, the same MobileNetV3-Small 1-frame model is repeatedly applied to each frame and the logits are averaged, so the parameter count is unchanged while inference cost increases with the number of frames.
- CNN-LSTM latency is measured for one 10-frame clip input with shape `[1, 10, 3, 224, 224]`.
- Peak GPU memory is reported only when CUDA is available. CPU-only runs use `N/A`.
- Single MobileNet per-frame latency: `{efficiency["per_frame_mobilenet_latency_ms"]}` ms.

## 1-frame vs 10-frame CNN-LSTM

- ACER reduction: `{acer_delta}` absolute.
- Latency increase: `{latency_delta}` ms.
- Peak GPU memory increase: `{memory_delta}` MB.

## Paper-Ready Sentences

1. The 1-frame model provides the simplest MobileNetV3-Small baseline using a single image without temporal evidence.
2. The 5-frame and 10-frame average variants increase multi-frame evidence without learning temporal order; they reuse the same MobileNetV3-Small backbone, so the parameter count is unchanged while inference cost increases with the number of evaluated frames.
3. The 10-frame CNN-LSTM adds order-aware temporal modeling by feeding 576-dimensional MobileNetV3-Small frame features into a one-layer LSTM with a 128-dimensional hidden state.
4. Relative to the 1-frame baseline, the 10-frame CNN-LSTM reduced test video-level ACER by `{acer_delta}` with a latency increase of `{latency_delta}` ms and a peak memory increase of `{memory_delta}` MB under the measured environment.
"""
    path.write_text(content)


def write_plan(path):
    content = """# Phase 1 Phase 2 Summary Implementation Plan

Goal: summarize the existing four Replay-PAD models without adding new experiments.

Architecture: read the existing result JSON files for performance, inspect the existing model definitions for structure, instantiate the two existing model classes for parameter and latency measurements, and write all paper-facing CSV/markdown artifacts under `outputs/results/`.

Tasks:
- Add focused tests for result row extraction, FP32 size calculation, and unknown-memory handling.
- Add `src/analysis/summarize_phase1_phase2.py` with CSV/markdown generation and latency/memory measurement.
- Run the script for the four requested models only: 1-frame, 5-frame avg, 10-frame avg, 10-frame CNN-LSTM.
- Verify the generated CSV and markdown outputs.
"""
    path.write_text(content)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    performance_rows, _ = build_performance_rows()
    efficiency = measure_efficiency(
        warmup=args.warmup,
        iterations=args.iterations,
    )
    efficiency_rows = build_efficiency_rows(performance_rows, efficiency)

    write_csv(RESULT_DIR / "model_performance_comparison.csv", performance_rows)
    write_csv(RESULT_DIR / "model_efficiency_comparison.csv", efficiency_rows)
    write_plan(RESULT_DIR / "phase1_phase2_summary_plan.md")
    write_rule_memo(RESULT_DIR / "devel_threshold_test_evaluation_rule.md")
    write_structure_memo(RESULT_DIR / "cnn_lstm_structure_memo.md")
    write_summary_markdown(
        RESULT_DIR / "phase1_phase2_summary.md",
        performance_rows,
        efficiency_rows,
        efficiency,
    )


if __name__ == "__main__":
    main()
