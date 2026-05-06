| Model | Backbone | Input type | Temporal length | Order used | Init | Accuracy | APCER | BPCER | ACER | Params | Size(MB) | Latency(ms) | Peak GPU Memory(MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN-LSTM | MobileNetV3-Small | 5-frame clip | 5 | Yes | Random | 0.995833 | 0.0 | 0.025 | 0.0125 | 1288738 | 4.916145 | 1.186081 | 26.587891 |
| MiniFASNet-LSTM | MiniFASNet | 5-frame clip | 5 | Yes | Random | 1.0 | 0.0 | 0.0 | 0.0 | 139994 | 0.534035 | 0.638516 | 41.314941 |
| MobileNetV4-LSTM | MobileNetV4 | 5-frame clip | 5 | Yes | Random | 1.0 | 0.0 | 0.0 | 0.0 | 490010 | 1.86924 | 0.688082 | 28.026855 |
| EfficientNet-Lite-LSTM | EfficientNet-Lite | 5-frame clip | 5 | Yes | Random | 0.995833 | 0.005 | 0.0 | 0.0025 | 420266 | 1.603188 | 2.171534 | 70.354492 |
| ShuffleNetV2-LSTM | ShuffleNetV2 | 5-frame clip | 5 | Yes | Random | 0.991667 | 0.0 | 0.05 | 0.025 | 1844710 | 7.03701 | 1.546491 | 35.675293 |
