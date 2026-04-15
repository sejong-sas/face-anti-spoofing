import torch
import torch.nn as nn
from torchvision import models


class CNNLSTMBinaryClassifier(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1, num_classes=2, pretrained=False):
        super().__init__()

        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None

        backbone = models.mobilenet_v3_small(weights=weights)
        self.feature_extractor = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feature_dim = 576  # MobileNetV3-Small final feature dim

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def extract_features(self, frames):
        feat = self.feature_extractor(frames)
        return self.avgpool(feat).flatten(1)

    def classify_features(self, feat):
        lstm_out, _ = self.lstm(feat)
        last_out = lstm_out[:, -1, :]          # 마지막 시점
        logits = self.classifier(last_out)
        return logits

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        feat = self.extract_features(x)        # [B*T, 576]
        feat = feat.view(b, t, -1)             # [B, T, 576]

        return self.classify_features(feat)

    def forward_streaming(self, x, chunk_size=1):
        # x: [B, T, C, H, W]
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        b, t, c, h, w = x.shape
        features = []

        for start in range(0, t, chunk_size):
            end = min(start + chunk_size, t)
            frames = x[:, start:end].contiguous().view(b * (end - start), c, h, w)
            chunk_feat = self.extract_features(frames)
            chunk_feat = chunk_feat.view(b, end - start, -1)
            features.append(chunk_feat)

        feat = torch.cat(features, dim=1)
        return self.classify_features(feat)
