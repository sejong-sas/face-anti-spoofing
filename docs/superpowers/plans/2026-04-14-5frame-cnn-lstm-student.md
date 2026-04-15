# 5-frame CNN-LSTM Student Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a supervised 5-frame CNN-LSTM student and compare its accuracy, memory, and latency against the current 5-frame average and 10-frame CNN-LSTM baselines.

**Architecture:** Reuse the existing `CNNLSTMBinaryClassifier` and `ReplayPADClipDataset`. Use `frame_index/replayattack_5frame_index.csv` as the clip dataset, with split filtering in the dataset loader. Extend evaluation metadata so clip5 results are labeled correctly. Add a focused summary script for efficiency/performance comparison.

**Tech Stack:** Python, PyTorch, pandas, existing Replay-Attack frame/clip indexes and PAD metric helpers.

---

### Task 1: Evaluation Metadata

**Files:**
- Modify: `src/engine/evaluate_cnn_lstm.py`
- Test: `tests/test_cnn_lstm_eval_metadata.py`

- [ ] Add a test that verifies result payload construction uses a caller-provided `input_type`.
- [ ] Run `pytest tests/test_cnn_lstm_eval_metadata.py -q`; expected failure is missing helper.
- [ ] Add `--input_type` argument and a helper that builds result/threshold payloads from args.
- [ ] Run the focused test; expected pass.

### Task 2: Train and Evaluate Student

**Files:**
- Create: `outputs/checkpoints/cnn_lstm_clip5_random_best.pth`
- Create: `outputs/results/cnn_lstm_clip5_eval_results.json`
- Create: `outputs/predictions/cnn_lstm_clip5_*`

- [ ] Run `python -m src.engine.train_cnn_lstm --train_csv /home/saslab01/Desktop/replay_pad/frame_index/replayattack_5frame_index.csv --devel_csv /home/saslab01/Desktop/replay_pad/frame_index/replayattack_5frame_index.csv --save_name cnn_lstm_clip5_random_best.pth --epochs 10 --batch_size 8`.
- [ ] Run `python -m src.engine.evaluate_cnn_lstm --devel_csv /home/saslab01/Desktop/replay_pad/frame_index/replayattack_5frame_index.csv --test_csv /home/saslab01/Desktop/replay_pad/frame_index/replayattack_5frame_index.csv --checkpoint_path /home/saslab01/Desktop/replay_pad/outputs/checkpoints/cnn_lstm_clip5_random_best.pth --tag cnn_lstm_clip5 --input_type "5-frame clip" --batch_size 8`.

### Task 3: Efficiency Summary

**Files:**
- Create: `src/analysis/summarize_clip5_student.py`
- Create: `outputs/results/cnn_lstm_clip5_efficiency_comparison.csv`
- Create: `outputs/results/cnn_lstm_clip5_student_summary.md`

- [ ] Add a script that reads existing 5-frame avg, 10-frame CNN-LSTM, and new 5-frame CNN-LSTM results.
- [ ] Measure params, FP32 size, latency, and peak GPU memory for 5-frame and 10-frame CNN-LSTM under the same measurement logic.
- [ ] Write CSV and markdown summary.
- [ ] Run focused tests and inspect generated outputs.
