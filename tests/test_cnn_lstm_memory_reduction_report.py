from src.analysis.compare_cnn_lstm_memory_reduction import (
    build_comparison_row,
    compute_delta_row,
)


def test_build_comparison_row_keeps_unknown_gpu_memory_uninvented():
    row = build_comparison_row(
        mode="streaming",
        metrics={
            "accuracy": 0.9958333333333333,
            "apcer": 0.0025,
            "bpcer": 0.0125,
            "acer": 0.0075,
        },
        threshold=0.018,
        latency_ms=3.25,
        peak_gpu_memory_mb=None,
    )

    assert row == {
        "Mode": "streaming",
        "Threshold": 0.018,
        "Accuracy": 0.995833,
        "APCER": 0.0025,
        "BPCER": 0.0125,
        "ACER": 0.0075,
        "Inference_Latency_ms": 3.25,
        "Peak_GPU_Memory_MB": "N/A",
    }


def test_compute_delta_row_reports_memory_reduction_and_latency_delta():
    baseline = {
        "Mode": "baseline",
        "ACER": 0.0075,
        "Inference_Latency_ms": 1.5,
        "Peak_GPU_Memory_MB": 46.0,
    }
    streaming = {
        "Mode": "streaming",
        "ACER": 0.0075,
        "Inference_Latency_ms": 2.5,
        "Peak_GPU_Memory_MB": 30.0,
    }

    row = compute_delta_row(baseline, streaming)

    assert row["Memory_Reduction_MB"] == 16.0
    assert row["Latency_Increase_ms"] == 1.0
    assert row["ACER_Delta"] == 0.0
