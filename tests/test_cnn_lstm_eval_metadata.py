from argparse import Namespace

from src.engine.evaluate_cnn_lstm import build_result_payload, build_threshold_payload


def test_build_payloads_use_requested_input_type():
    args = Namespace(tag="cnn_lstm_clip5", input_type="5-frame clip")
    devel_metrics = {"acer": 0.1}
    test_metrics = {"acer": 0.2}

    threshold_payload = build_threshold_payload(args, 0.123, devel_metrics)
    result_payload = build_result_payload(
        args=args,
        threshold=0.123,
        devel_metrics=devel_metrics,
        test_metrics=test_metrics,
        artifacts={"test_video_predictions_csv": "pred.csv"},
    )

    assert threshold_payload["input_type"] == "5-frame clip"
    assert result_payload["input_type"] == "5-frame clip"
    assert result_payload["test_video_metrics"] == test_metrics
