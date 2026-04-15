# CNN-LSTM Memory Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a memory-reduced streaming inference path for the existing 10-frame CNN-LSTM and measure its memory/latency/performance trade-off.

**Architecture:** Keep the existing CNN-LSTM checkpoint and classification path. Add a feature extraction mode that processes clip frames in chunks before feeding the same LSTM/classifier. Add a comparison script that evaluates baseline and streaming inference on the existing 10-frame clip predictions pipeline and writes CSV/markdown outputs under `outputs/results/`.

**Tech Stack:** Python, PyTorch, pandas, existing ReplayPAD clip dataset and PAD metric helpers.

---

### Task 1: Streaming Forward Path

**Files:**
- Modify: `src/models/cnn_lstm_baseline.py`
- Test: `tests/test_cnn_lstm_streaming.py`

- [ ] Add a test that copies weights between two CNN-LSTM instances and verifies normal forward and streaming forward logits match on a small eval-mode tensor.
- [ ] Run `pytest tests/test_cnn_lstm_streaming.py -q`; expected failure is an `AttributeError` for missing streaming forward support.
- [ ] Add `extract_features`, `classify_features`, and `forward_streaming` methods to `CNNLSTMBinaryClassifier`.
- [ ] Run `pytest tests/test_cnn_lstm_streaming.py -q`; expected pass.

### Task 2: Memory Reduction Comparison Script

**Files:**
- Create: `src/analysis/compare_cnn_lstm_memory_reduction.py`
- Test: `tests/test_cnn_lstm_memory_reduction_report.py`

- [ ] Add tests for comparison row construction and unknown GPU memory handling.
- [ ] Run `pytest tests/test_cnn_lstm_memory_reduction_report.py -q`; expected failure is missing module.
- [ ] Implement the comparison script to evaluate baseline and streaming inference on devel/test clip10, select threshold on devel, evaluate test video-level metrics, measure latency and peak memory, and write CSV/markdown outputs.
- [ ] Run the focused tests; expected pass.

### Task 3: Generate Outputs

**Files:**
- Create: `outputs/results/cnn_lstm_memory_reduction_comparison.csv`
- Create: `outputs/results/cnn_lstm_memory_reduction_summary.md`

- [ ] Run `python -m src.analysis.compare_cnn_lstm_memory_reduction --warmup 5 --iterations 20`.
- [ ] Inspect generated CSV and markdown to confirm both baseline and streaming rows exist.
- [ ] Run focused tests again and report the measured memory/latency/performance trade-off.
