# CNN-LSTM Memory Reduction Design

## Goal

Reduce peak GPU memory for the existing 10-frame CNN-LSTM experiment while preserving test video-level performance as much as possible. Latency is recorded as a secondary trade-off metric.

## Scope

The experiment uses only the existing 10-frame CNN-LSTM architecture and checkpoint. It does not introduce a new dataset, new backbone, attention model, 20-frame model, shuffled/reversed-frame analysis, or pretrained initialization.

## Design

The baseline CNN-LSTM reshapes a clip from `[B, T, C, H, W]` to `[B*T, C, H, W]` and sends all frames through MobileNetV3-Small features in one pass. This can increase peak activation memory. The memory-reduced variant keeps the same weights and classification path but extracts MobileNet features frame-by-frame or in small chunks, then stacks the resulting `[B, 576]` features into `[B, T, 576]` before running the same LSTM and classifier.

## Outputs

- `outputs/results/cnn_lstm_memory_reduction_comparison.csv`
- `outputs/results/cnn_lstm_memory_reduction_summary.md`

## Success Criteria

- Existing checkpoint loads into both baseline and streaming inference paths.
- Baseline and streaming logits match closely for the same input in eval mode.
- The comparison script reports Accuracy, APCER, BPCER, ACER, latency, peak GPU memory, memory delta, and latency delta.
- If CUDA is unavailable, peak memory is reported as `N/A` rather than invented.
