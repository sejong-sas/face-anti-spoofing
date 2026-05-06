from argparse import Namespace

from src.engine.evaluate_temporal_lightweight_lstm import (
    build_result_payload,
    build_threshold_payload,
    sanitize_metrics,
)


def test_eval_payloads_exclude_hter():
    args = Namespace(backbone="minifasnet", input_type="5-frame clip")
    devel_metrics = {"accuracy": 1.0, "apcer": 0.0, "bpcer": 0.0, "acer": 0.0, "hter": 0.0}
    test_metrics = {"accuracy": 0.9, "apcer": 0.1, "bpcer": 0.2, "acer": 0.15, "hter": 0.15}

    threshold_payload = build_threshold_payload(args, 0.123, devel_metrics)
    result_payload = build_result_payload(args, 0.123, devel_metrics, test_metrics, artifacts={})

    assert "hter" not in threshold_payload["devel_metrics"]
    assert "hter" not in result_payload["devel_threshold_search_result"]
    assert "hter" not in result_payload["test_video_metrics"]
    assert sanitize_metrics(test_metrics)["acer"] == 0.15

