# Temporal Lightweight Backbone Summary

## 1. Experiment Purpose
Same 5-frame temporal-order condition, backbone only changes, to compare the performance-efficiency trade-off.

## 2. Experimental Conditions
- Replay-Attack only
- 5-frame clip input
- temporal order preserved through an LSTM head
- random initialization only
- threshold chosen on devel video-level scores only
- test evaluated at video level with the fixed devel threshold
- HTER excluded completely

## 3. Performance Comparison
| Model | Backbone | Input type | Temporal length | Order used | Init | Accuracy | APCER | BPCER | ACER | Params | Size(MB) | Latency(ms) | Peak GPU Memory(MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN-LSTM | MobileNetV3-Small | 5-frame clip | 5 | Yes | Random | 0.995833 | 0.0 | 0.025 | 0.0125 | 1288738 | 4.916145 | 1.186081 | 26.587891 |
| MiniFASNet-LSTM | MiniFASNet | 5-frame clip | 5 | Yes | Random | 1.0 | 0.0 | 0.0 | 0.0 | 139994 | 0.534035 | 0.638516 | 41.314941 |
| MobileNetV4-LSTM | MobileNetV4 | 5-frame clip | 5 | Yes | Random | 1.0 | 0.0 | 0.0 | 0.0 | 490010 | 1.86924 | 0.688082 | 28.026855 |
| EfficientNet-Lite-LSTM | EfficientNet-Lite | 5-frame clip | 5 | Yes | Random | 0.995833 | 0.005 | 0.0 | 0.0025 | 420266 | 1.603188 | 2.171534 | 70.354492 |
| ShuffleNetV2-LSTM | ShuffleNetV2 | 5-frame clip | 5 | Yes | Random | 0.991667 | 0.0 | 0.05 | 0.025 | 1844710 | 7.03701 | 1.546491 | 35.675293 |

Accuracy, APCER, BPCER, and ACER are reported at the video level. ACER is the primary detection metric for interpretation.

## 4. Efficiency Comparison
Latency(ms) is the forward-pass time for one `[1, 5, 3, 224, 224]` clip. Peak GPU Memory(MB) is the peak forward-pass allocation when CUDA is available.

## 5. Conclusions
- Most accurate model: MiniFASNet-LSTM
- Lowest ACER: MiniFASNet-LSTM
- Fastest model: MiniFASNet-LSTM
- Lowest memory model: CNN-LSTM
- Best trade-off model: MobileNetV4-LSTM

## 6. Baseline Comparison
The first row is the existing 5-frame CNN-LSTM baseline using MobileNetV3-Small.

## 8. Commands
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

python -m src.analysis.compare_temporal_lightweight_efficiency \
  --results_dir outputs/results/upgrade \
  --checkpoints_dir outputs/checkpoints/upgrade \
  --output_csv outputs/results/upgrade/temporal_lightweight_backbone_comparison.csv \
  --output_md outputs/results/upgrade/temporal_lightweight_backbone_comparison.md \
  --summary_md outputs/results/upgrade/temporal_lightweight_backbone_summary.md
```
