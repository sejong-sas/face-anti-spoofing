# 5-frame CNN-LSTM Student Summary

Device: `cuda`

## Comparison

| Model | Input type | Temporal length | Threshold | Accuracy | APCER | BPCER | ACER | Params | FP32_Model_Size_MB | Inference_Latency_ms | Peak_GPU_Memory_MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5-frame CNN-LSTM student | 5-frame clip | 5 | 0.103 | 0.995833 | 0.0 | 0.025 | 0.0125 | 1288738 | 4.916145 | 1.219988 | 32.423828 |
| 10-frame CNN-LSTM | 10-frame clip | 10 | 0.018 | 0.995833 | 0.0025 | 0.0125 | 0.0075 | 1288738 | 4.916145 | 1.810784 | 44.050781 |

## Delta vs 10-frame CNN-LSTM

| Student | Teacher | ACER_Delta_vs_10frame_CNN_LSTM | Latency_Reduction_ms | Memory_Reduction_MB |
| --- | --- | --- | --- | --- |
| 5-frame CNN-LSTM student | 10-frame CNN-LSTM | 0.005 | 0.590796 | 11.626953 |

## Reference vs 5-frame avg

- 5-frame avg ACER: `0.0375`; 5-frame CNN-LSTM student ACER: `0.0125`.
- 5-frame avg latency: `5.527589` ms; 5-frame CNN-LSTM student latency: `1.219988` ms.
- 5-frame avg peak memory: `28.282227` MB; 5-frame CNN-LSTM student peak memory: `32.423828` MB.


Interpretation: the 5-frame CNN-LSTM student keeps order-aware temporal modeling while halving the frame budget relative to the 10-frame CNN-LSTM. The main question is whether its ACER remains close to the 10-frame CNN-LSTM while reducing latency and peak GPU memory.
