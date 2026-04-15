# Phase 1 Phase 2 Summary Implementation Plan

Goal: summarize the existing four Replay-PAD models without adding new experiments.

Architecture: read the existing result JSON files for performance, inspect the existing model definitions for structure, instantiate the two existing model classes for parameter and latency measurements, and write all paper-facing CSV/markdown artifacts under `outputs/results/`.

Tasks:
- Add focused tests for result row extraction, FP32 size calculation, and unknown-memory handling.
- Add `src/analysis/summarize_phase1_phase2.py` with CSV/markdown generation and latency/memory measurement.
- Run the script for the four requested models only: 1-frame, 5-frame avg, 10-frame avg, 10-frame CNN-LSTM.
- Verify the generated CSV and markdown outputs.
