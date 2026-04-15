# Phase 1 and Phase 2 Summary

## Target Files

- Model definitions: `src/models/mobilenetv3_small_baseline.py`, `src/models/cnn_lstm_baseline.py`
- Evaluation scripts: `src/engine/evaluate_image_1frame.py`, `src/engine/evaluate_image_5frame_avg.py`, `src/engine/evaluate_image_10frame_avg.py`, `src/engine/evaluate_cnn_lstm.py`
- Result JSON files: `outputs/results/mobilenetv3_small_1frame_random_eval_results.json`, `outputs/results/mobilenetv3_small_5frame_avg_eval_results.json`, `outputs/results/mobilenetv3_small_10frame_avg_eval_results.json`, `outputs/results/cnn_lstm_clip10_eval_results.json`
- Prediction CSV files: `outputs/predictions/image_1frame_random_*`, `outputs/predictions/image_5frame_avg_*`, `outputs/predictions/image_10frame_avg_*`, `outputs/predictions/cnn_lstm_clip10_*`

## Performance Table

| Model | Input type | Temporal length | Order used | Initialization | Threshold source | Threshold | Accuracy | APCER | BPCER | ACER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1-frame | single frame | 1 | No | random | devel | 0.121 | 0.9875 | 0.0 | 0.075 | 0.0375 |
| 5-frame avg | 5-frame average | 5 | No | random | devel | 0.116 | 0.9875 | 0.0 | 0.075 | 0.0375 |
| 10-frame avg | 10-frame average | 10 | No | random | devel | 0.123 | 0.9875 | 0.0 | 0.075 | 0.0375 |
| 10-frame CNN-LSTM | 10-frame clip | 10 | Yes | random | devel | 0.018 | 0.995833 | 0.0025 | 0.0125 | 0.0075 |

## Efficiency Table

Measured device: `cuda`

| Model | Input type | Temporal length | Params | FP32_Model_Size_MB | Inference_Latency_ms | Per_Frame_Latency_ms | Latency_Unit | Peak_GPU_Memory_MB | Threshold | Accuracy | APCER | BPCER | ACER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1-frame | single frame | 1 | 1519906 | 5.797981 | 1.12263 | 1.12263 | per video/sequence | 28.280273 | 0.121 | 0.9875 | 0.0 | 0.075 | 0.0375 |
| 5-frame avg | 5-frame average | 5 | 1519906 | 5.797981 | 5.527589 | 1.12263 | per video/sequence | 28.282227 | 0.116 | 0.9875 | 0.0 | 0.075 | 0.0375 |
| 10-frame avg | 10-frame average | 10 | 1519906 | 5.797981 | 10.989778 | 1.12263 | per video/sequence | 28.284668 | 0.123 | 0.9875 | 0.0 | 0.075 | 0.0375 |
| 10-frame CNN-LSTM | 10-frame clip | 10 | 1288738 | 4.916145 | 1.58702 | N/A | per video/sequence | 46.366211 | 0.018 | 0.995833 | 0.0025 | 0.0125 | 0.0075 |

## Efficiency Measurement Notes

- Parameter count is the number of trainable and non-trainable parameters in the instantiated PyTorch model.
- FP32 model size is computed as `Params * 4 / 1024 / 1024`.
- Inference latency is measured with dummy tensors at 224x224 resolution after warmup, using `torch.no_grad()`. `Inference_Latency_ms` is the per-video or per-sequence value.
- `Per_Frame_Latency_ms` is reported for the MobileNetV3-Small frame model and its frame-averaging variants. It is `N/A` for CNN-LSTM because the measured unit is the whole 10-frame clip.
- For 5-frame avg and 10-frame avg, the same MobileNetV3-Small 1-frame model is repeatedly applied to each frame and the logits are averaged, so the parameter count is unchanged while inference cost increases with the number of frames.
- CNN-LSTM latency is measured for one 10-frame clip input with shape `[1, 10, 3, 224, 224]`.
- Peak GPU memory is reported only when CUDA is available. CPU-only runs use `N/A`.
- Single MobileNet per-frame latency: `1.12263` ms.

## 1-frame vs 10-frame CNN-LSTM

- ACER reduction: `0.03` absolute.
- Latency increase: `0.46439` ms.
- Peak GPU memory increase: `18.085938` MB.

## Paper-Ready Sentences

1. The 1-frame model provides the simplest MobileNetV3-Small baseline using a single image without temporal evidence.
2. The 5-frame and 10-frame average variants increase multi-frame evidence without learning temporal order; they reuse the same MobileNetV3-Small backbone, so the parameter count is unchanged while inference cost increases with the number of evaluated frames.
3. The 10-frame CNN-LSTM adds order-aware temporal modeling by feeding 576-dimensional MobileNetV3-Small frame features into a one-layer LSTM with a 128-dimensional hidden state.
4. Relative to the 1-frame baseline, the 10-frame CNN-LSTM reduced test video-level ACER by `0.03` with a latency increase of `0.46439` ms and a peak memory increase of `18.085938` MB under the measured environment.
