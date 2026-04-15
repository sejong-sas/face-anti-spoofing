import csv
import json
import time
from pathlib import Path

import torch

from src.models.cnn_lstm_baseline import CNNLSTMBinaryClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO_ROOT / "outputs" / "results"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints"


def rounded(value, digits=6):
    return round(float(value), digits)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fp32_size_mb(params):
    return rounded(params * 4 / (1024 * 1024))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def build_student_row(
    model,
    input_type,
    temporal_length,
    result,
    params,
    fp32_size_mb,
    latency_ms,
    peak_gpu_memory_mb,
):
    metrics = result["test_video_metrics"]
    return {
        "Model": model,
        "Input type": input_type,
        "Temporal length": temporal_length,
        "Threshold": rounded(result["threshold"]),
        "Accuracy": rounded(metrics["accuracy"]),
        "APCER": rounded(metrics["apcer"]),
        "BPCER": rounded(metrics["bpcer"]),
        "ACER": rounded(metrics["acer"]),
        "Params": int(params),
        "FP32_Model_Size_MB": rounded(fp32_size_mb),
        "Inference_Latency_ms": rounded(latency_ms),
        "Peak_GPU_Memory_MB": rounded(peak_gpu_memory_mb)
        if peak_gpu_memory_mb is not None
        else "N/A",
    }


def compute_student_deltas(student, teacher):
    memory_reduction = "N/A"
    if student["Peak_GPU_Memory_MB"] != "N/A" and teacher["Peak_GPU_Memory_MB"] != "N/A":
        memory_reduction = rounded(
            teacher["Peak_GPU_Memory_MB"] - student["Peak_GPU_Memory_MB"]
        )

    return {
        "Student": student["Model"] if "Model" in student else "5-frame CNN-LSTM student",
        "Teacher": teacher["Model"] if "Model" in teacher else "10-frame CNN-LSTM",
        "ACER_Delta_vs_10frame_CNN_LSTM": rounded(student["ACER"] - teacher["ACER"]),
        "Latency_Reduction_ms": rounded(
            teacher["Inference_Latency_ms"] - student["Inference_Latency_ms"]
        ),
        "Memory_Reduction_MB": memory_reduction,
    }


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_latency_ms(model, temporal_length, device, warmup=5, iterations=20):
    clip = torch.randn(1, temporal_length, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(clip)
        synchronize_if_cuda(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(clip)
        synchronize_if_cuda(device)
        end = time.perf_counter()
    return (end - start) * 1000 / iterations


def measure_peak_memory_mb(model, temporal_length, device):
    if device.type != "cuda":
        return None
    clip = torch.randn(1, temporal_length, 3, 224, 224, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        model(clip)
    synchronize_if_cuda(device)
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def load_cnn_lstm(checkpoint_path, device):
    model = CNNLSTMBinaryClassifier(
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


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


def write_summary(path, rows, deltas, device):
    five_frame_avg = load_reference_row(
        RESULT_DIR / "model_efficiency_comparison.csv",
        "5-frame avg",
    )
    reference_text = ""
    if five_frame_avg is not None:
        student = rows[0]
        reference_text = f"""
## Reference vs 5-frame avg

- 5-frame avg ACER: `{five_frame_avg["ACER"]}`; 5-frame CNN-LSTM student ACER: `{student["ACER"]}`.
- 5-frame avg latency: `{five_frame_avg["Inference_Latency_ms"]}` ms; 5-frame CNN-LSTM student latency: `{student["Inference_Latency_ms"]}` ms.
- 5-frame avg peak memory: `{five_frame_avg["Peak_GPU_Memory_MB"]}` MB; 5-frame CNN-LSTM student peak memory: `{student["Peak_GPU_Memory_MB"]}` MB.
"""

    content = f"""# 5-frame CNN-LSTM Student Summary

Device: `{device}`

## Comparison

{markdown_table(rows)}

## Delta vs 10-frame CNN-LSTM

{markdown_table([deltas])}
{reference_text}

Interpretation: the 5-frame CNN-LSTM student keeps order-aware temporal modeling while halving the frame budget relative to the 10-frame CNN-LSTM. The main question is whether its ACER remains close to the 10-frame CNN-LSTM while reducing latency and peak GPU memory.
"""
    path.write_text(content)


def load_reference_row(path, model_name):
    if not path.exists():
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Model") == model_name:
                return row
    return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip5_result = load_json(RESULT_DIR / "cnn_lstm_clip5_eval_results.json")
    clip10_result = load_json(RESULT_DIR / "cnn_lstm_clip10_eval_results.json")

    clip5_model = load_cnn_lstm(
        CHECKPOINT_DIR / "cnn_lstm_clip5_random_best.pth",
        device,
    )
    clip10_model = load_cnn_lstm(
        CHECKPOINT_DIR / "cnn_lstm_clip10_random_best.pth",
        device,
    )

    params = count_params(clip5_model)
    rows = [
        build_student_row(
            model="5-frame CNN-LSTM student",
            input_type="5-frame clip",
            temporal_length=5,
            result=clip5_result,
            params=params,
            fp32_size_mb=fp32_size_mb(params),
            latency_ms=measure_latency_ms(clip5_model, 5, device),
            peak_gpu_memory_mb=measure_peak_memory_mb(clip5_model, 5, device),
        ),
        build_student_row(
            model="10-frame CNN-LSTM",
            input_type="10-frame clip",
            temporal_length=10,
            result=clip10_result,
            params=count_params(clip10_model),
            fp32_size_mb=fp32_size_mb(count_params(clip10_model)),
            latency_ms=measure_latency_ms(clip10_model, 10, device),
            peak_gpu_memory_mb=measure_peak_memory_mb(clip10_model, 10, device),
        ),
    ]
    deltas = compute_student_deltas(rows[0], rows[1])

    write_csv(RESULT_DIR / "cnn_lstm_clip5_efficiency_comparison.csv", rows)
    write_csv(RESULT_DIR / "cnn_lstm_clip5_student_delta.csv", [deltas])
    write_summary(
        RESULT_DIR / "cnn_lstm_clip5_student_summary.md",
        rows,
        deltas,
        device,
    )


if __name__ == "__main__":
    main()
