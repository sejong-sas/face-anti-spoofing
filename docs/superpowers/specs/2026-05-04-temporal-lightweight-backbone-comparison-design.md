# Temporal Lightweight Backbone Comparison Design

**Goal:** Compare lightweight CNN backbones under the same 5-frame temporal-order Replay-Attack setting, using an LSTM temporal head and video-level evaluation.

**Scope:** This experiment reuses the existing Replay-Attack 5-frame clip index, the same train/devel/test split convention already used by the project, the same 5-frame CNN-LSTM training and evaluation rules, and the same video-level thresholding protocol. The only variable is the CNN backbone paired with the same LSTM head.

**Constraints:** No pretrained weights, no external checkpoints, no HTER anywhere in outputs or summaries, and no frame-level-only conclusions.

---

## Problem Statement

The repository already contains a 5-frame CNN-LSTM baseline built on MobileNetV3-Small. The new experiment asks whether alternative lightweight CNN backbones can improve the trade-off between video-level spoof detection quality and inference efficiency when the temporal setup stays fixed.

The core question is:

> Under the same 5-frame temporal-order condition, which lightweight backbone gives the best performance-efficiency trade-off?

## Experimental Conditions

- Dataset: Replay-Attack only.
- Input: 5-frame clips with shape `[B, 5, 3, 224, 224]`.
- Temporal modeling: one LSTM with `hidden_dim=128` and `num_layers=1`.
- Initialization: random initialization only.
- Thresholding: select threshold on devel video-level scores only.
- Test: apply the devel-selected threshold without modification.
- Evaluation: video-level only.
- Metrics kept: Accuracy, APCER, BPCER, ACER.
- Metrics excluded: HTER.

## Models

1. CNN-LSTM
   - Backbone: MobileNetV3-Small
   - Baseline reference using the existing implementation and checkpoint when available.

2. MiniFASNet-LSTM
   - Backbone: MiniFASNet

3. MobileNetV4-LSTM
   - Backbone: MobileNetV4
   - If the current environment cannot construct this backbone without pretrained weights, record the failure reason and skip the model.

4. EfficientNet-Lite-LSTM
   - Backbone: EfficientNet-Lite
   - If the current environment cannot construct this backbone without pretrained weights, record the failure reason and skip the model.

5. ShuffleNetV2-LSTM
   - Backbone: ShuffleNetV2

## Data Flow

1. Load the 5-frame index CSV and filter by split inside the dataset loader.
2. Read each clip as five RGB frames and resize them to `224x224`.
3. Pass each frame through the same backbone to produce a sequence of frame features.
4. Feed the feature sequence into an LSTM.
5. Classify the final LSTM output into binary logits.
6. Convert clip predictions to video predictions by averaging clip scores per `video_id`.
7. Select the best threshold on devel video-level scores.
8. Apply that threshold to test video-level scores.

## Output Artifacts

- Checkpoints:
  - `outputs/checkpoints/upgrade/{backbone}_lstm_clip5_random_best.pth`
- Evaluation JSON:
  - `outputs/results/upgrade/{backbone}_lstm_clip5_eval_results.json`
- Comparison table:
  - `outputs/results/upgrade/temporal_lightweight_backbone_comparison.csv`
  - `outputs/results/upgrade/temporal_lightweight_backbone_comparison.md`
- Summary:
  - `outputs/results/upgrade/temporal_lightweight_backbone_summary.md`

## Existing Baseline Reuse

- Existing 5-frame CNN-LSTM result file:
  - `outputs/results/cnn_lstm_clip5_eval_results.json`
- Existing 5-frame index:
  - `frame_index/replayattack_5frame_index.csv`
- Existing checkpoint:
  - Use `outputs/checkpoints/cnn_lstm_clip5_random_best.pth` if it exists in the current repo.
  - If not present, fall back to the adjacent `../replay_pad/outputs/checkpoints/cnn_lstm_clip5_random_best.pth`.

## Implementation Notes

- Use a single shared model class that wraps backbone selection, feature pooling, and LSTM classification.
- Backbone selection should avoid loading pretrained weights.
- Feature dimensionality must be inferred or explicitly handled per backbone so the LSTM input size is always correct.
- Efficiency measurements should use dummy inputs and measure only model forward cost, not decoding or preprocessing.
- Peak GPU memory should be measured only when CUDA is available.

## Acceptance Criteria

- Every runnable model produces a `[B, 2]` logits tensor for a `[B, 5, 3, 224, 224]` input.
- Evaluation outputs contain only the requested metrics and no HTER field.
- The comparison table includes the baseline CNN-LSTM first, followed by the lightweight backbones that successfully trained and evaluated.
- If MobileNetV4 or EfficientNet-Lite cannot be implemented in the local environment, the summary records the reason and the rest of the experiment still completes.

