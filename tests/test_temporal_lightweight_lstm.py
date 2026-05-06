import torch

from src.models.temporal_lightweight_lstm import TemporalBackboneLSTM


def test_supported_backbones_forward_shape():
    for backbone in [
        "mobilenetv3_small",
        "minifasnet",
        "mobilenetv4_small",
        "efficientnet_lite",
        "shufflenetv2",
    ]:
        model = TemporalBackboneLSTM(
            backbone_name=backbone,
            hidden_dim=128,
            num_layers=1,
            num_classes=2,
            pretrained=False,
        )
        x = torch.randn(2, 5, 3, 224, 224)
        y = model(x)
        assert y.shape == (2, 2)


def test_pretrained_true_is_rejected():
    try:
        TemporalBackboneLSTM(
            backbone_name="mobilenetv3_small",
            hidden_dim=128,
            num_layers=1,
            num_classes=2,
            pretrained=True,
        )
    except ValueError as exc:
        assert "pretrained=True is not allowed" in str(exc)
    else:
        raise AssertionError("Expected pretrained=True to be rejected")

