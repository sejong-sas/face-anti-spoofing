import torch

from src.datasets.replay_pad_clip_dataset import ReplayPADClipDataset
from src.models.cnn_lstm_baseline import CNNLSTMBinaryClassifier


def test_clip_dataset_parses_pipe_separated_frame_paths():
    dataset = ReplayPADClipDataset.__new__(ReplayPADClipDataset)

    paths = dataset._parse_frame_paths("a.jpg|b.jpg|c.jpg")

    assert paths == ["a.jpg", "b.jpg", "c.jpg"]


def test_streaming_forward_matches_batched_forward_in_eval_mode():
    torch.manual_seed(7)
    model = CNNLSTMBinaryClassifier(
        hidden_dim=16,
        num_layers=1,
        num_classes=2,
        pretrained=False,
    )
    model.eval()

    clips = torch.randn(1, 3, 3, 64, 64)

    with torch.no_grad():
        batched_logits = model(clips)
        streaming_logits = model.forward_streaming(clips, chunk_size=1)

    assert torch.allclose(batched_logits, streaming_logits, atol=1e-6)
