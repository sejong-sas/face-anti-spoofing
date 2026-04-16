# 5-frame CNN-LSTM Error Analysis

## Scope

- Model: 5-frame CNN-LSTM student
- Evaluation split: Replay-Attack test
- Threshold source: devel
- Threshold: 0.103
- Video-level rule: mean clip score per video, then thresholding

## Main Findings

The 5-frame CNN-LSTM made 2 video-level errors on the test split. Both errors are false positives from real/adverse videos of the same subject:

| Video | Label | Prediction | Score | Error |
|---|---:|---:|---:|---|
| test__real__real__client014_session01_webcam_authenticate_adverse_1 | real | attack | 0.852429 | FP |
| test__real__real__client014_session01_webcam_authenticate_adverse_2 | real | attack | 0.974687 | FP |

No attack videos were missed at the devel-selected threshold, so the current weakness is not APCER. The weakness is BPCER: a small subset of genuine videos under adverse illumination is pushed into the attack region.

## Clip-Level Behavior

The two false-positive videos are not caused by one or two outlier clips. All 16 clips in each video exceed the threshold.

| Video | Clip min | Clip max | Clip std | Clips over threshold |
|---|---:|---:|---:|---:|
| adverse_1 | 0.527562 | 0.988363 | 0.162401 | 16 / 16 |
| adverse_2 | 0.939318 | 0.992302 | 0.017864 | 16 / 16 |

Interpretation:

- `adverse_1` is unstable across time, but even its lowest clip score is far above 0.103.
- `adverse_2` is consistently classified as attack across the whole video.
- Mean aggregation cannot recover these videos because every clip already looks attack-like to the model.

## Threshold Stability

The devel split is perfectly separated at the selected threshold:

| Split | Max real score | Min attack score | Threshold |
|---|---:|---:|---:|
| devel | 0.102234 | 0.801749 | 0.103 |
| test | 0.974687 | 0.757971 | 0.103 |

The devel threshold is technically valid, but the margin above the highest devel real video is very small: `0.103 - 0.102234 = 0.000766`. The real issue is generalization: the test split contains real/adverse samples whose scores are much higher than any real sample seen in devel.

Post-hoc threshold changes on test are not valid for final reporting, but they show the trade-off:

| Threshold | FP | FN | APCER | BPCER | ACER |
|---:|---:|---:|---:|---:|---:|
| 0.103 | 2 | 0 | 0.0000 | 0.0250 | 0.0125 |
| 0.900 | 1 | 2 | 0.0050 | 0.0125 | 0.0088 |
| 0.975 | 0 | 5 | 0.0125 | 0.0000 | 0.0063 |

This suggests the model has a score calibration/generalization problem rather than a simple threshold-selection bug.

## Visual/Low-Level Diagnostics

A contact sheet was saved to:

- `/home/saslab01/Desktop/replay_pad/outputs/analysis/clip5_error_analysis/client014_real_contact_sheet.jpg`

Compared with the same client's controlled real videos, the false-positive adverse videos have a different background/window pattern and illumination condition. Simple frame diagnostics also differ:

| Video group | Mean brightness | Mean contrast | Mean saturation | Mean sharpness |
|---|---:|---:|---:|---:|
| client014 adverse real FP | 0.549-0.556 | 0.300-0.304 | 0.497-0.498 | 0.00336-0.00340 |
| client014 controlled real correct | 0.542-0.548 | 0.225 | 0.755-0.759 | 0.00114-0.00119 |

The FP videos are higher contrast and sharper than the controlled real videos, and their saturation statistics are very different. This does not prove causality, but it supports the hypothesis that the student is using appearance/illumination cues that do not generalize cleanly to adverse genuine samples.

## Where The Model Is Vulnerable

1. Genuine adverse videos from specific subjects/environments
   - Actual errors are concentrated in `client014`, `real`, `adverse`.
   - Both adverse real videos for this client are false positives.
   - The controlled real videos for the same client are correctly classified with near-zero score.

2. Clip score instability under temporal/appearance variation
   - `adverse_1` has high clip score variance: 0.162401.
   - Several correctly classified attack videos also have high variance, especially hand/mobile attacks.
   - The model can classify these correctly, but the score trace is less stable.

3. Calibration gap between devel and test
   - Devel real scores stop at 0.102234.
   - Test real scores include 0.852429 and 0.974687.
   - This means the model's devel threshold is valid under the implemented rule, but devel does not cover this failure mode.

## Feature Ideas To Improve Robustness

These are feature/experiment ideas, not implemented changes:

1. Add a lightweight per-clip score stability feature at video aggregation time.
   - Use mean plus variance/min/max or a robust aggregation rule.
   - Goal: detect videos where the model is uncertain or temporally inconsistent.
   - Limitation: the worst FP (`adverse_2`) is consistently high, so aggregation alone will not fully solve the issue.

2. Add illumination/quality-aware auxiliary features.
   - Candidate features: brightness, contrast, saturation, sharpness, frame-to-frame difference.
   - Use them as analysis features first, then consider a small calibration head or stratified normalization.
   - Goal: reduce false attack confidence on adverse real videos.

3. Strengthen real/adverse representation in training or validation.
   - The current failure is BPCER on real/adverse, not APCER.
   - A targeted setting would evaluate whether real/adverse augmentation or balanced sampling lowers false positives without hurting attack recall.

4. Use teacher-guided 5-frame student calibration.
   - Keep the 5-frame student for memory/latency efficiency.
   - Add distillation or calibration against the 10-frame teacher score, especially on real/adverse videos.
   - Goal: keep the current efficiency gain while reducing overconfident false positives.

## Generated Files

- `outputs/analysis/clip5_error_analysis/clip5_focus_video_diagnostics.csv`
- `outputs/analysis/clip5_error_analysis/clip5_video_score_stability.csv`
- `outputs/analysis/clip5_error_analysis/clip5_fp_clip_score_trace.csv`
- `outputs/analysis/clip5_error_analysis/client014_real_contact_sheet.jpg`
- `outputs/analysis/clip5_error_analysis/clip5_error_analysis_summary.md`
