# CNN-LSTM Memory Reduction Summary

Device: `cuda`

Streaming chunk size: `1`

## Comparison

| Mode | Threshold | Accuracy | APCER | BPCER | ACER | Inference_Latency_ms | Peak_GPU_Memory_MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.018 | 0.995833 | 0.0025 | 0.0125 | 0.0075 | 1.647319 | 39.029297 |
| streaming | 0.018 | 0.995833 | 0.0025 | 0.0125 | 0.0075 | 11.152765 | 29.445801 |

## Delta

| Baseline_Mode | Streaming_Mode | Memory_Reduction_MB | Latency_Increase_ms | ACER_Delta |
| --- | --- | --- | --- | --- |
| baseline | streaming | 9.583496 | 9.505446 | 0.0 |

Interpretation: streaming inference keeps the same CNN-LSTM checkpoint and classification path, but extracts MobileNetV3-Small frame features in smaller chunks before the LSTM. The goal is to reduce peak GPU memory while preserving test video-level ACER; latency is recorded as the expected trade-off.
