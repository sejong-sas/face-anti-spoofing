from src.analysis.compare_temporal_lightweight_efficiency import build_row


def test_comparison_row_columns_match_required_schema():
    row = build_row(
        model_name="CNN-LSTM",
        backbone="MobileNetV3-Small",
        input_type="5-frame clip",
        result_json={
            "test_video_metrics": {
                "accuracy": 0.99,
                "apcer": 0.01,
                "bpcer": 0.02,
                "acer": 0.015,
            }
        },
        params=1_000_000,
        latency_ms=3.14,
        peak_mem_mb=12.5,
    )

    expected_keys = [
        "Model",
        "Backbone",
        "Input type",
        "Temporal length",
        "Order used",
        "Init",
        "Accuracy",
        "APCER",
        "BPCER",
        "ACER",
        "Params",
        "Size(MB)",
        "Latency(ms)",
        "Peak GPU Memory(MB)",
    ]
    assert list(row.keys()) == expected_keys
    assert "HTER" not in row

