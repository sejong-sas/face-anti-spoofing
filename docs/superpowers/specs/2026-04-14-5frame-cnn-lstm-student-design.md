# 5-frame CNN-LSTM Student Design

## Goal

Train and evaluate a supervised 5-frame CNN-LSTM student to test whether order-aware temporal modeling can retain most of the 10-frame CNN-LSTM accuracy while reducing peak memory and latency.

## Scope

This experiment adds one model setting: `5-frame CNN-LSTM student`. It reuses the existing MobileNetV3-Small feature extractor, one-layer LSTM, Replay-Attack 5-frame index, devel threshold selection, and test video-level evaluation rule. It does not add pretrained weights, attention, 20-frame analysis, shuffled/reversed order analysis, or a teacher-distillation loss in this first pass.

## Design

Training uses `frame_index/replayattack_5frame_index.csv` for both train and devel splits, filtered by the dataset's `split` column. The model is the existing `CNNLSTMBinaryClassifier` with `hidden_dim=128`, `num_layers=1`, and random initialization. Evaluation uses the same 5-frame index for devel and test splits, selects the threshold on devel video-level scores, and applies the fixed threshold on test video-level scores.

## Outputs

- `outputs/checkpoints/cnn_lstm_clip5_random_best.pth`
- `outputs/results/cnn_lstm_clip5_eval_results.json`
- `outputs/predictions/cnn_lstm_clip5_*`
- `outputs/results/cnn_lstm_clip5_efficiency_comparison.csv`
- `outputs/results/cnn_lstm_clip5_student_summary.md`

## Success Criteria

- The 5-frame CNN-LSTM student is compared against 5-frame avg and 10-frame CNN-LSTM.
- Peak GPU memory and latency are measured for 5-frame CNN-LSTM and 10-frame CNN-LSTM under the same measurement logic.
- The summary reports whether ACER is preserved, whether memory decreases, and whether latency decreases.
