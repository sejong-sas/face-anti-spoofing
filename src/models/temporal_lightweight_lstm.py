import torch
import torch.nn as nn
from torchvision import models


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    if name == "hardswish":
        return nn.Hardswish(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, activation="relu"):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            _make_activation(activation),
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.gate(scale)
        return x * scale


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation="relu"):
        super().__init__()
        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            activation=activation,
        )
        self.pointwise = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation,
        )

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, activation="silu", use_se=True):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvBNAct(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    activation=activation,
                )
            )
        layers.append(
            ConvBNAct(
                hidden_dim if expand_ratio != 1 else in_channels,
                hidden_dim if expand_ratio != 1 else in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim if expand_ratio != 1 else in_channels,
                activation=activation,
            )
        )
        self.body = nn.Sequential(*layers)
        self.se = SqueezeExcitation(hidden_dim if expand_ratio != 1 else in_channels) if use_se else nn.Identity()
        self.project = nn.Sequential(
            nn.Conv2d(
                hidden_dim if expand_ratio != 1 else in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.body(x)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return out


class FusedMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_channels=None, activation="hardswish"):
        super().__init__()
        hidden = expand_channels if expand_channels is not None else out_channels
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            ConvBNAct(
                in_channels,
                hidden,
                kernel_size=3,
                stride=stride,
                padding=1,
                activation=activation,
            ),
            ConvBNAct(
                hidden,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                activation=activation,
            ),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class MiniFASNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 24, kernel_size=3, stride=2, padding=1, activation="relu"),
            DepthwiseSeparableBlock(24, 24, stride=1, activation="relu"),
            DepthwiseSeparableBlock(24, 32, stride=2, activation="relu"),
            DepthwiseSeparableBlock(32, 32, stride=1, activation="relu"),
            DepthwiseSeparableBlock(32, 48, stride=2, activation="relu"),
            DepthwiseSeparableBlock(48, 48, stride=1, activation="relu"),
            DepthwiseSeparableBlock(48, 64, stride=2, activation="relu"),
            DepthwiseSeparableBlock(64, 64, stride=1, activation="relu"),
            ConvBNAct(64, 96, kernel_size=1, stride=1, padding=0, activation="relu"),
        )

    def forward(self, x):
        return self.features(x)


class MobileNetV4SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 16, kernel_size=3, stride=2, padding=1, activation="hardswish"),
            FusedMBConvBlock(16, 24, stride=2, expand_channels=32, activation="hardswish"),
            FusedMBConvBlock(24, 24, stride=1, expand_channels=32, activation="hardswish"),
            FusedMBConvBlock(24, 40, stride=2, expand_channels=48, activation="hardswish"),
            FusedMBConvBlock(40, 40, stride=1, expand_channels=48, activation="hardswish"),
            InvertedResidualBlock(40, 64, stride=2, expand_ratio=4, activation="silu", use_se=True),
            InvertedResidualBlock(64, 96, stride=1, expand_ratio=4, activation="silu", use_se=True),
            InvertedResidualBlock(96, 128, stride=1, expand_ratio=4, activation="silu", use_se=True),
            ConvBNAct(128, 160, kernel_size=1, stride=1, padding=0, activation="hardswish"),
        )

    def forward(self, x):
        return self.features(x)


class EfficientNetLiteBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 24, kernel_size=3, stride=2, padding=1, activation="silu"),
            InvertedResidualBlock(24, 24, stride=1, expand_ratio=1, activation="silu", use_se=True),
            InvertedResidualBlock(24, 32, stride=2, expand_ratio=4, activation="silu", use_se=True),
            InvertedResidualBlock(32, 32, stride=1, expand_ratio=4, activation="silu", use_se=True),
            InvertedResidualBlock(32, 48, stride=2, expand_ratio=4, activation="silu", use_se=True),
            InvertedResidualBlock(48, 48, stride=1, expand_ratio=4, activation="silu", use_se=True),
            InvertedResidualBlock(48, 64, stride=2, expand_ratio=4, activation="silu", use_se=True),
            InvertedResidualBlock(64, 96, stride=1, expand_ratio=6, activation="silu", use_se=True),
            ConvBNAct(96, 128, kernel_size=1, stride=1, padding=0, activation="silu"),
        )

    def forward(self, x):
        return self.features(x)


def _build_shufflenetv2_features():
    backbone = models.shufflenet_v2_x1_0(weights=None)
    return nn.Sequential(
        backbone.conv1,
        backbone.maxpool,
        backbone.stage2,
        backbone.stage3,
        backbone.stage4,
        backbone.conv5,
    )


def _build_torchvision_mobilenetv3_small():
    backbone = models.mobilenet_v3_small(weights=None)
    return backbone.features


def _build_backbone(backbone_name: str):
    name = backbone_name.lower()
    if name == "mobilenetv3_small":
        return _build_torchvision_mobilenetv3_small()
    if name == "minifasnet":
        return MiniFASNetBackbone()
    if name == "mobilenetv4_small":
        return MobileNetV4SmallBackbone()
    if name == "efficientnet_lite":
        return EfficientNetLiteBackbone()
    if name == "shufflenetv2":
        return _build_shufflenetv2_features()
    raise ValueError(
        "Unsupported backbone_name: "
        f"{backbone_name}. Expected one of mobilenetv3_small, minifasnet, "
        "mobilenetv4_small, efficientnet_lite, shufflenetv2."
    )


class TemporalBackboneLSTM(nn.Module):
    def __init__(
        self,
        backbone_name,
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        pretrained=False,
    ):
        super().__init__()

        if pretrained:
            raise ValueError("pretrained=True is not allowed for this experiment")

        self.backbone_name = backbone_name.lower()
        self.feature_extractor = _build_backbone(self.backbone_name)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = self._infer_feature_dim()

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _infer_feature_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.feature_extractor(dummy)
            pooled = self.avgpool(features).flatten(1)
        return int(pooled.shape[1])

    def extract_features(self, frames):
        features = self.feature_extractor(frames)
        return self.avgpool(features).flatten(1)

    def classify_features(self, feature_seq):
        lstm_out, _ = self.lstm(feature_seq)
        last_out = lstm_out[:, -1, :]
        return self.classifier(last_out)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected clip tensor with shape [B, T, C, H, W], got {tuple(x.shape)}")

        batch_size, temporal_length, channels, height, width = x.shape
        frames = x.reshape(batch_size * temporal_length, channels, height, width)
        frame_features = self.extract_features(frames)
        feature_seq = frame_features.reshape(batch_size, temporal_length, -1)
        return self.classify_features(feature_seq)

