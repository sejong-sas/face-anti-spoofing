# CNN-LSTM Structure Memo

The CNN-LSTM model uses MobileNetV3-Small as a frame-level feature extractor. In the current code, the model instantiates `torchvision.models.mobilenet_v3_small`, keeps only `backbone.features`, and does not use the original MobileNetV3 classifier head.

For each input clip `x` with shape `[B, T, C, H, W]`, the frames are reshaped to `[B*T, C, H, W]` and passed through `backbone.features`. Adaptive average pooling maps the final convolutional feature map to one vector per frame. The resulting frame-level feature dimension is 576.

The temporal module is a one-layer LSTM with `input_size=576`, `hidden_size=128`, and `batch_first=True`. The model uses the LSTM output at the last time step as the clip representation, and a linear classifier maps this 128-dimensional vector to two logits for binary real/spoof classification.

Paper-ready wording: We used MobileNetV3-Small up to its convolutional feature extractor as the per-frame backbone and removed the original classifier head. Each frame was represented by a 576-dimensional pooled feature vector, and the sequence of 10 frame features was passed to a one-layer LSTM with a 128-dimensional hidden state. The final hidden output was classified by a linear layer into real and spoof classes.
