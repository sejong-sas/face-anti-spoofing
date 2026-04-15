from src.analysis.summarize_clip5_student import build_student_row, compute_student_deltas


def test_build_student_row_combines_metrics_and_efficiency():
    row = build_student_row(
        model="5-frame CNN-LSTM student",
        input_type="5-frame clip",
        temporal_length=5,
        result={
            "threshold": 0.103,
            "test_video_metrics": {
                "accuracy": 0.9958333333333333,
                "apcer": 0.0,
                "bpcer": 0.025,
                "acer": 0.0125,
            },
        },
        params=1_288_738,
        fp32_size_mb=4.916145,
        latency_ms=2.5,
        peak_gpu_memory_mb=33.0,
    )

    assert row["Model"] == "5-frame CNN-LSTM student"
    assert row["Temporal length"] == 5
    assert row["ACER"] == 0.0125
    assert row["Peak_GPU_Memory_MB"] == 33.0


def test_compute_student_deltas_against_10frame_teacher():
    student = {
        "ACER": 0.0125,
        "Inference_Latency_ms": 2.0,
        "Peak_GPU_Memory_MB": 30.0,
    }
    teacher = {
        "ACER": 0.0075,
        "Inference_Latency_ms": 4.0,
        "Peak_GPU_Memory_MB": 40.0,
    }

    deltas = compute_student_deltas(student, teacher)

    assert deltas["ACER_Delta_vs_10frame_CNN_LSTM"] == 0.005
    assert deltas["Latency_Reduction_ms"] == 2.0
    assert deltas["Memory_Reduction_MB"] == 10.0
