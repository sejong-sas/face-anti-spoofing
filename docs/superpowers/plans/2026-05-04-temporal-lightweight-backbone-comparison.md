# Temporal Lightweight Backbone Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build shared temporal-backbone LSTM training, evaluation, and efficiency tooling to compare lightweight CNN backbones under the same 5-frame Replay-Attack protocol.

**Architecture:** Add one shared model wrapper that turns a named CNN backbone into a frame feature extractor, then runs a fixed LSTM temporal head and binary classifier. Reuse the existing ReplayPAD clip dataset and video-level metric helpers so training and evaluation stay aligned with the current CNN-LSTM baseline. Add one comparison script that loads the baseline and any new checkpoints, measures size/latency/memory with dummy inputs, and writes CSV/Markdown summaries without HTER.

**Tech Stack:** Python, PyTorch, torchvision, optional timm for fallback backbone construction, pandas, argparse, CSV/Markdown writers, existing project dataset and metric helpers.

---

### Task 1: Confirm paths and baseline artifacts

**Files:**
- Inspect: `frame_index/replayattack_5frame_index.csv`
- Inspect: `outputs/results/cnn_lstm_clip5_eval_results.json`
- Inspect: `outputs/checkpoints/cnn_lstm_clip5_random_best.pth` if present
- Inspect fallback path: `../replay_pad/outputs/checkpoints/cnn_lstm_clip5_random_best.pth`

- [ ] **Step 1: Verify the 5-frame index exists in the current repo**

Run:
```bash
ls -l frame_index/replayattack_5frame_index.csv
```

Expected:
```text
frame_index/replayattack_5frame_index.csv
```

- [ ] **Step 2: Verify the current-repo checkpoint path and the fallback adjacent path**

Run:
```bash
ls -l outputs/checkpoints/cnn_lstm_clip5_random_best.pth
ls -l ../replay_pad/outputs/checkpoints/cnn_lstm_clip5_random_best.pth
```

Expected:
- At least one file exists.
- The implementation should prefer the current repo path when both exist.

- [ ] **Step 3: Verify the baseline result JSON exists**

Run:
```bash
ls -l outputs/results/cnn_lstm_clip5_eval_results.json
```

Expected:
```text
outputs/results/cnn_lstm_clip5_eval_results.json
```

### Task 2: Add the shared temporal backbone model

**Files:**
- Create: `src/models/temporal_lightweight_lstm.py`
- Modify: `src/models/cnn_lstm_baseline.py` only if you choose to reuse helpers there, but keep the baseline model behavior unchanged
- Test: `tests/test_temporal_lightweight_lstm.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from src.models.temporal_lightweight_lstm import TemporalBackboneLSTM

def test_supported_backbones_forward_shape():
    for backbone in ["mobilenetv3_small", "minifasnet", "shufflenetv2", "efficientnet_lite"]:
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
pytest tests/test_temporal_lightweight_lstm.py -q
```

Expected:
- Import failure because the module does not exist yet.

- [ ] **Step 3: Implement backbone adapters and temporal head**

Implement:
```python
class TemporalBackboneLSTM(nn.Module):
    def __init__(self, backbone_name, hidden_dim=128, num_layers=1, num_classes=2, pretrained=False):
        ...
```

Behavior:
- Build a feature extractor for each frame.
- Remove classifier heads.
- Pool to a 1D feature vector per frame.
- Infer `feature_dim`.
- Feed `[B, 5, D]` into LSTM.
- Return `[B, 2]` logits.
- Never load pretrained weights.

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
pytest tests/test_temporal_lightweight_lstm.py -q
```

Expected:
- `mobilenetv3_small torch.Size([2, 2])`
- `minifasnet torch.Size([2, 2])`
- `shufflenetv2 torch.Size([2, 2])`
- `efficientnet_lite torch.Size([2, 2])`

- [ ] **Step 5: Add a MobileNetV4 construction test if supported locally**

```python
import pytest
import torch
from src.models.temporal_lightweight_lstm import TemporalBackboneLSTM

def test_mobilenetv4_optional():
    try:
        model = TemporalBackboneLSTM(
            backbone_name="mobilenetv4_small",
            hidden_dim=128,
            num_layers=1,
            num_classes=2,
            pretrained=False,
        )
    except Exception as exc:
        pytest.skip(f"mobilenetv4_small unavailable: {exc}")
    x = torch.randn(1, 5, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 2)
```

### Task 3: Add training script for temporal lightweight backbones

**Files:**
- Create: `src/engine/train_temporal_lightweight_lstm.py`
- Test: `tests/test_train_temporal_lightweight_lstm_smoke.py`

- [ ] **Step 1: Write a smoke test that exercises argument parsing and one short training step**

Use a tiny subset or a mocked dataloader if needed, but keep the same public CLI:
```python
from src.engine.train_temporal_lightweight_lstm import parse_args
```

- [ ] **Step 2: Run the smoke test and verify it fails before implementation**

Run:
```bash
pytest tests/test_train_temporal_lightweight_lstm_smoke.py -q
```

Expected:
- Module import failure before the file exists.

- [ ] **Step 3: Implement the training CLI**

Required CLI flags:
```python
--backbone
--clip_csv
--epochs
--batch_size
--lr
--img_size
--seed
--save_path
```

Training rules:
- Use `ReplayPADClipDataset`.
- Filter by split internally.
- Train on `train`, validate on `devel`.
- Save the best checkpoint by devel accuracy.
- Use `Adam` and `CrossEntropyLoss`.
- Use random initialization only.

- [ ] **Step 4: Run the smoke test**

Run:
```bash
pytest tests/test_train_temporal_lightweight_lstm_smoke.py -q
```

Expected:
- CLI parses.
- A short run saves a checkpoint.

### Task 4: Add evaluation script for temporal lightweight backbones

**Files:**
- Create: `src/engine/evaluate_temporal_lightweight_lstm.py`
- Test: `tests/test_evaluate_temporal_lightweight_lstm_smoke.py`

- [ ] **Step 1: Write a smoke test that checks JSON shape and metric keys**

The test should assert the JSON contains:
- `model`
- `input_type`
- `initialization`
- `threshold`
- `devel_threshold_search_result`
- `test_video_metrics`
- `artifacts`

It should assert there is no `hter` key anywhere in the top-level payload.

- [ ] **Step 2: Run the smoke test and confirm failure before implementation**

Run:
```bash
pytest tests/test_evaluate_temporal_lightweight_lstm_smoke.py -q
```

- [ ] **Step 3: Implement devel-threshold selection and video-level test evaluation**

Requirements:
- Load a checkpoint.
- Predict clip scores for devel and test.
- Aggregate to video level by `video_id` mean score.
- Select threshold on devel video scores only.
- Apply that threshold to test video scores.
- Save JSON to `outputs/results/upgrade/{backbone}_lstm_clip5_eval_results.json`.
- Exclude HTER entirely.

- [ ] **Step 4: Run the smoke test**

Run:
```bash
pytest tests/test_evaluate_temporal_lightweight_lstm_smoke.py -q
```

Expected:
- JSON is written and metric keys are correct.

### Task 5: Add efficiency comparison script

**Files:**
- Create: `src/analysis/compare_temporal_lightweight_efficiency.py`
- Test: `tests/test_compare_temporal_lightweight_efficiency_smoke.py`

- [ ] **Step 1: Write a smoke test for the comparison table shape**

The test should validate the CSV header contains:
- `Model`
- `Backbone`
- `Input type`
- `Temporal length`
- `Order used`
- `Init`
- `Accuracy`
- `APCER`
- `BPCER`
- `ACER`
- `Params`
- `Size(MB)`
- `Latency(ms)`
- `Peak GPU Mem(MB)`

It should assert there is no `HTER` column.

- [ ] **Step 2: Run the smoke test and confirm failure before implementation**

Run:
```bash
pytest tests/test_compare_temporal_lightweight_efficiency_smoke.py -q
```

- [ ] **Step 3: Implement model loading, size computation, latency, and peak memory measurement**

Requirements:
- Use dummy input `[1, 5, 3, 224, 224]`.
- Measure forward-pass latency only.
- Use CUDA synchronization when applicable.
- Compute peak memory with `reset_peak_memory_stats()` and `max_memory_allocated()`.
- Read evaluation JSON files and merge them with measured efficiency metrics.
- Include the baseline CNN-LSTM row first.

- [ ] **Step 4: Run the smoke test**

Run:
```bash
pytest tests/test_compare_temporal_lightweight_efficiency_smoke.py -q
```

### Task 6: Run the actual experiment for the supported backbones

**Files:**
- Use: `frame_index/replayattack_5frame_index.csv`
- Output checkpoints: `outputs/checkpoints/upgrade/*.pth`
- Output results: `outputs/results/upgrade/*.json`

- [ ] **Step 1: Train and evaluate MiniFASNet-LSTM**

Run:
```bash
python -m src.engine.train_temporal_lightweight_lstm \
  --backbone minifasnet \
  --clip_csv frame_index/replayattack_5frame_index.csv \
  --epochs 10 \
  --batch_size 8 \
  --lr 0.0001 \
  --img_size 224 \
  --seed 42 \
  --save_path outputs/checkpoints/upgrade/minifasnet_lstm_clip5_random_best.pth

python -m src.engine.evaluate_temporal_lightweight_lstm \
  --backbone minifasnet \
  --checkpoint outputs/checkpoints/upgrade/minifasnet_lstm_clip5_random_best.pth \
  --clip_csv frame_index/replayattack_5frame_index.csv \
  --output_json outputs/results/upgrade/minifasnet_lstm_clip5_eval_results.json
```

- [ ] **Step 2: Train and evaluate ShuffleNetV2-LSTM**

Run:
```bash
python -m src.engine.train_temporal_lightweight_lstm \
  --backbone shufflenetv2 \
  --clip_csv frame_index/replayattack_5frame_index.csv \
  --epochs 10 \
  --batch_size 8 \
  --lr 0.0001 \
  --img_size 224 \
  --seed 42 \
  --save_path outputs/checkpoints/upgrade/shufflenetv2_lstm_clip5_random_best.pth

python -m src.engine.evaluate_temporal_lightweight_lstm \
  --backbone shufflenetv2 \
  --checkpoint outputs/checkpoints/upgrade/shufflenetv2_lstm_clip5_random_best.pth \
  --clip_csv frame_index/replayattack_5frame_index.csv \
  --output_json outputs/results/upgrade/shufflenetv2_lstm_clip5_eval_results.json
```

- [ ] **Step 3: Train and evaluate EfficientNet-Lite-LSTM if the backbone can be constructed locally**

Run:
```bash
python -m src.engine.train_temporal_lightweight_lstm \
  --backbone efficientnet_lite \
  --clip_csv frame_index/replayattack_5frame_index.csv \
  --epochs 10 \
  --batch_size 8 \
  --lr 0.0001 \
  --img_size 224 \
  --seed 42 \
  --save_path outputs/checkpoints/upgrade/efficientnet_lite_lstm_clip5_random_best.pth

python -m src.engine.evaluate_temporal_lightweight_lstm \
  --backbone efficientnet_lite \
  --checkpoint outputs/checkpoints/upgrade/efficientnet_lite_lstm_clip5_random_best.pth \
  --clip_csv frame_index/replayattack_5frame_index.csv \
  --output_json outputs/results/upgrade/efficientnet_lite_lstm_clip5_eval_results.json
```

If construction fails, record the exception message in the summary and continue with the remaining models.

- [ ] **Step 4: Train and evaluate MobileNetV4-LSTM if the backbone can be constructed locally**

Run:
```bash
python -m src.engine.train_temporal_lightweight_lstm \
  --backbone mobilenetv4_small \
  --clip_csv frame_index/replayattack_5frame_index.csv \
  --epochs 10 \
  --batch_size 8 \
  --lr 0.0001 \
  --img_size 224 \
  --seed 42 \
  --save_path outputs/checkpoints/upgrade/mobilenetv4_small_lstm_clip5_random_best.pth

python -m src.engine.evaluate_temporal_lightweight_lstm \
  --backbone mobilenetv4_small \
  --checkpoint outputs/checkpoints/upgrade/mobilenetv4_small_lstm_clip5_random_best.pth \
  --clip_csv frame_index/replayattack_5frame_index.csv \
  --output_json outputs/results/upgrade/mobilenetv4_small_lstm_clip5_eval_results.json
```

If construction fails, record the exception message in the summary and continue.

### Task 7: Generate the comparison table and summary

**Files:**
- Create: `outputs/results/upgrade/temporal_lightweight_backbone_comparison.csv`
- Create: `outputs/results/upgrade/temporal_lightweight_backbone_comparison.md`
- Create: `outputs/results/upgrade/temporal_lightweight_backbone_summary.md`

- [ ] **Step 1: Implement the merge logic with the baseline result as the first row**

Required baseline row:
- Model: `CNN-LSTM`
- Backbone: `MobileNetV3-Small`
- Use `outputs/results/cnn_lstm_clip5_eval_results.json`
- Use current-repo checkpoint if present, otherwise fallback to `../replay_pad`

- [ ] **Step 2: Run the comparison script**

Run:
```bash
python -m src.analysis.compare_temporal_lightweight_efficiency \
  --results_dir outputs/results/upgrade \
  --checkpoints_dir outputs/checkpoints/upgrade \
  --output_csv outputs/results/upgrade/temporal_lightweight_backbone_comparison.csv \
  --output_md outputs/results/upgrade/temporal_lightweight_backbone_comparison.md \
  --summary_md outputs/results/upgrade/temporal_lightweight_backbone_summary.md
```

- [ ] **Step 3: Write the summary with the required interpretation**

The summary must cover:
- experimental purpose
- experimental conditions
- performance comparison using Accuracy, APCER, BPCER, ACER
- efficiency comparison using Params, Size(MB), Latency(ms), Peak GPU Memory(MB)
- conclusion naming the best accuracy, best ACER, fastest, lowest-memory, and best trade-off model
- explanation of whether any backbone was skipped and why
- no HTER references

### Task 8: Verify outputs and final consistency

**Files:**
- Verify all generated files under `outputs/checkpoints/upgrade/` and `outputs/results/upgrade/`

- [ ] **Step 1: Verify checkpoint files exist**

Run:
```bash
find outputs/checkpoints/upgrade -maxdepth 1 -type f | sort
```

Expected:
- One checkpoint per successfully trained model.

- [ ] **Step 2: Verify evaluation JSON files exist**

Run:
```bash
find outputs/results/upgrade -maxdepth 1 -name '*_eval_results.json' | sort
```

Expected:
- One JSON per successfully evaluated model.

- [ ] **Step 3: Verify comparison artifacts exist**

Run:
```bash
ls -l outputs/results/upgrade/temporal_lightweight_backbone_comparison.csv
ls -l outputs/results/upgrade/temporal_lightweight_backbone_comparison.md
ls -l outputs/results/upgrade/temporal_lightweight_backbone_summary.md
```

Expected:
- All three files exist.

- [ ] **Step 4: Verify no HTER appears in outputs**

Run:
```bash
rg -n "HTER|hter" outputs/results/upgrade outputs/results/cnn_lstm_clip5_eval_results.json
```

Expected:
- No matches in the newly generated outputs.

