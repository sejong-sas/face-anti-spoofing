import math

from src.analysis.summarize_phase1_phase2 import (
    build_efficiency_row,
    build_performance_row,
    fp32_model_size_mb,
)


def test_build_performance_row_uses_test_video_metrics_and_devel_threshold():
    result = {
        "model": "MobileNetV3-Small",
        "input_type": "10-frame average",
        "initialization": "random",
        "threshold_selected_on": "devel",
        "threshold": 0.123,
        "test_video_metrics": {
            "accuracy": 0.9875,
            "apcer": 0.0,
            "bpcer": 0.075,
            "acer": 0.0375,
        },
    }

    row = build_performance_row(
        name="10-frame avg",
        result=result,
        input_type="10-frame average",
        temporal_length=10,
        order_used="No",
        threshold_source="devel",
    )

    assert row == {
        "Model": "10-frame avg",
        "Input type": "10-frame average",
        "Temporal length": 10,
        "Order used": "No",
        "Initialization": "random",
        "Threshold source": "devel",
        "Threshold": 0.123,
        "Accuracy": 0.9875,
        "APCER": 0.0,
        "BPCER": 0.075,
        "ACER": 0.0375,
    }


def test_fp32_model_size_mb_is_derived_from_parameter_count():
    assert math.isclose(fp32_model_size_mb(1_048_576), 4.0)


def test_build_efficiency_row_keeps_unknown_memory_uninvented():
    perf_row = {
        "Model": "10-frame CNN-LSTM",
        "Input type": "10-frame clip",
        "Temporal length": 10,
        "Threshold": 0.018,
        "Accuracy": 0.9958333333333333,
        "APCER": 0.0025,
        "BPCER": 0.0125,
        "ACER": 0.0075,
    }

    row = build_efficiency_row(
        perf_row=perf_row,
        params=1_000,
        latency_ms=12.34,
        per_frame_latency_ms=None,
        peak_gpu_memory_mb=None,
    )

    assert row["Params"] == 1_000
    assert row["FP32_Model_Size_MB"] == fp32_model_size_mb(1_000)
    assert row["Inference_Latency_ms"] == 12.34
    assert row["Per_Frame_Latency_ms"] == "N/A"
    assert row["Latency_Unit"] == "per video/sequence"
    assert row["Peak_GPU_Memory_MB"] == "N/A"
