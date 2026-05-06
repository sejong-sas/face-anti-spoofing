# 5-frame Temporal Lightweight Backbone 비교 실험 보고서

## 1. 실험 배경과 목적

기존 ASK 제출 논문에서는 Replay-Attack 데이터셋에서 `5-frame CNN-LSTM`을 사용했고, backbone은 `MobileNetV3-Small`이었다. 이번 후속 실험은 이 기존 구조를 완전히 바꾸는 것이 아니라, 발표 준비와 향후 단말기 적용 가능성 검토를 위해 `5-frame temporal order` 조건은 그대로 유지한 채 backbone만 바꿔서 성능과 효율성의 균형을 더 자세히 비교한 것이다.

쉽게 말하면, 이 실험은 “5장의 프레임을 순서대로 보는 방식”은 그대로 두고, “각 프레임에서 특징을 뽑는 backbone”만 바꿔본 실험이다. 여기서 backbone은 프레임 한 장에서 얼굴의 패턴과 질감을 추출하는 부분이고, LSTM은 5장의 순서와 시간 흐름을 반영하는 부분이다. 따라서 이번 실험의 핵심은 “최고 정확도 모델 하나를 찾는 것”이 아니라, 같은 5-frame temporal 조건에서 어떤 경량 backbone이 가장 좋은 성능-효율 trade-off를 보이는지 확인하는 것이다.

핵심 질문은 다음과 같다.

> 같은 5-frame temporal order 조건에서 어떤 경량 backbone이 가장 좋은 성능-효율 trade-off를 보이는가?

이번 실험은 교수님 피드백처럼 단말기 적용 가능성을 더 강하게 보여주기 위한 후속 연구 성격을 가진다. 즉, 기존 논문의 방향을 뒤집는 것이 아니라, 발표와 랩미팅에서 “왜 이 구조가 괜찮은지”를 더 설득력 있게 보강하기 위한 비교 실험이다.

## 2. 비교한 모델과 선정 이유

| Model | Backbone | 선정 이유 | 이번 실험에서의 공통 구조 |
| --- | --- | --- | --- |
| CNN-LSTM | MobileNetV3-Small | 기존 ASK 논문의 5-frame 기준 모델 | 5-frame clip + LSTM |
| MiniFASNet-LSTM | MiniFASNet | 얼굴 생체인식/anti-spoofing에서 경량 구조로 자주 쓰이는 계열 | 5-frame clip + LSTM |
| MobileNetV4-LSTM | MobileNetV4 | 모바일/엣지 환경을 고려한 최신 경량 MobileNet 계열 | 5-frame clip + LSTM |
| EfficientNet-Lite-LSTM | EfficientNet-Lite | 모바일 환경에 맞춘 EfficientNet 계열 경량 모델 | 5-frame clip + LSTM |
| ShuffleNetV2-LSTM | ShuffleNetV2 | 속도와 메모리 효율을 고려한 대표적인 경량 CNN | 5-frame clip + LSTM |

모든 모델은 backbone만 다르고, 입력 프레임 수와 temporal head는 완전히 동일하게 맞췄다. 따라서 모델 간 차이는 frame-level feature extractor의 차이로 해석해야 한다. 즉, 모델마다 입력 프레임 수나 평가 방식이 달라서 생기는 불공정한 비교가 아니라, backbone 선택 자체의 영향을 비교한 것이다.

또한 `Order used = Yes`로 통일했다. 이는 5장의 프레임을 아무 순서나 섞지 않고 시간 순서를 유지해 LSTM이 temporal information을 보게 했다는 뜻이다.

## 3. 실험 조건

| 항목 | 설정 |
| --- | --- |
| Dataset | Replay-Attack |
| Input type | 5-frame clip |
| Input shape | `[B, 5, 3, 224, 224]` |
| Temporal order | Yes |
| Temporal module | LSTM |
| LSTM hidden_dim | 128 |
| LSTM num_layers | 1 |
| Initialization | Random |
| Pretrained | 사용 안 함 |
| Epochs | 10 |
| Batch size | 8 |
| Learning rate | 0.0001 |
| Image size | 224 |
| Seed | 42 |
| Threshold selection | devel split |
| Final evaluation | test split, video-level |
| Performance metrics | Accuracy, APCER, BPCER, ACER |
| Efficiency metrics | Params, Size(MB), Latency(ms), Peak GPU Memory(MB) |
| HTER | 제외 |

모든 모델은 동일한 5-frame clip 입력과 동일한 LSTM 구조를 사용하였으며, 차이는 frame-level feature extractor인 backbone에만 있다.

중요한 점은 다음과 같다.

- threshold는 devel split의 video-level score에서만 선택했다.
- test 결과를 보고 threshold나 hyperparameter를 바꾸지 않았다.
- 최종 평가는 반드시 video-level 기준으로만 수행했다.
- pretrained weight는 전혀 사용하지 않았고, 모든 모델은 random initialization으로 학습했다.

## 4. 코드 구현 방식

### 4.1 공통 모델 코드

파일: [src/models/temporal_lightweight_lstm.py](/home/saslab01/Desktop/replay-attack/src/models/temporal_lightweight_lstm.py)

이 파일에는 `TemporalBackboneLSTM` 클래스를 구현했다. 이 클래스의 역할은 아주 단순하게 말하면 다음과 같다.

1. 5-frame clip을 입력으로 받는다.
2. 각 frame을 backbone에 넣어서 feature vector를 만든다.
3. 5개의 feature vector를 시간 순서대로 LSTM에 넣는다.
4. LSTM의 마지막 출력으로 real/spoof를 분류한다.

즉, 모델 구조는 “5장의 사진을 한 장씩 보고 특징을 뽑은 다음, 그 특징들을 순서대로 이어서 최종 판단하는 방식”이다. 여기서 backbone은 사진 한 장에서 특징을 뽑는 역할이고, LSTM은 5장의 순서를 보는 역할이다.

이 파일에서 중요한 점은 backbone_name만 바꾸면 같은 temporal head를 유지한 채 여러 경량 CNN 계열을 비교할 수 있다는 것이다. 이번 실험에서는 `mobilenetv3_small`, `minifasnet`, `mobilenetv4_small`, `efficientnet_lite`, `shufflenetv2`를 같은 방식으로 연결했다.

또한 pretrained=True는 코드에서 명시적으로 거부되도록 만들어, 실험 조건이 어긋나지 않게 했다.

추가로 중요한 해석 포인트가 하나 있다. 이번 구현은 외부 pretrained checkpoint를 쓰지 않기 때문에, 실제 실험은 “각 backbone family를 random initialization으로 구성한 경량 feature extractor” 비교라고 보는 것이 정확하다. 즉, backbone family의 아이디어를 유지하되, pretrained 기반의 성능 증폭은 배제했다.

### 4.2 학습 코드

파일: [src/engine/train_temporal_lightweight_lstm.py](/home/saslab01/Desktop/replay-attack/src/engine/train_temporal_lightweight_lstm.py)

이 파일은 backbone별 학습을 수행한다. 핵심 흐름은 다음과 같다.

- `--backbone` 인자로 어떤 backbone을 학습할지 선택한다.
- `ReplayPADClipDataset`을 사용해서 Replay-Attack 5-frame clip을 읽는다.
- train split으로 학습하고 devel split으로 검증한다.
- devel accuracy가 가장 높을 때 checkpoint를 저장한다.
- 최종 checkpoint는 `outputs/checkpoints/upgrade/` 아래에 저장된다.

학습 조건은 모두 동일했다.

- epochs = 10
- batch size = 8
- learning rate = 0.0001
- seed = 42
- optimizer = Adam
- loss = CrossEntropyLoss
- pretrained = False

저장된 checkpoint 예시는 다음과 같다.

- [outputs/checkpoints/upgrade/minifasnet_lstm_clip5_random_best.pth](/home/saslab01/Desktop/replay-attack/outputs/checkpoints/upgrade/minifasnet_lstm_clip5_random_best.pth)
- [outputs/checkpoints/upgrade/shufflenetv2_lstm_clip5_random_best.pth](/home/saslab01/Desktop/replay-attack/outputs/checkpoints/upgrade/shufflenetv2_lstm_clip5_random_best.pth)
- [outputs/checkpoints/upgrade/efficientnet_lite_lstm_clip5_random_best.pth](/home/saslab01/Desktop/replay-attack/outputs/checkpoints/upgrade/efficientnet_lite_lstm_clip5_random_best.pth)
- [outputs/checkpoints/upgrade/mobilenetv4_small_lstm_clip5_random_best.pth](/home/saslab01/Desktop/replay-attack/outputs/checkpoints/upgrade/mobilenetv4_small_lstm_clip5_random_best.pth)

### 4.3 평가 코드

파일: [src/engine/evaluate_temporal_lightweight_lstm.py](/home/saslab01/Desktop/replay-attack/src/engine/evaluate_temporal_lightweight_lstm.py)

이 파일은 학습된 checkpoint를 불러온 뒤, devel과 test에서 clip score를 계산하고 video-level 평가를 수행한다.

핵심 흐름은 다음과 같다.

1. checkpoint를 로드한다.
2. devel split에서 clip score를 계산한다.
3. test split에서도 clip score를 계산한다.
4. 같은 `video_id`를 가진 clip score들을 평균내어 video-level score를 만든다.
5. devel video-level score에서 threshold를 선택한다.
6. 선택한 threshold를 test split에 고정 적용한다.
7. Accuracy, APCER, BPCER, ACER를 계산한다.

비전공자용으로 풀면, clip 하나마다 점수가 나오지만 최종 판단은 영상 하나 단위로 해야 하므로 같은 영상에서 나온 clip 점수들을 평균내어 video-level score를 만든 것이다. 그리고 threshold는 시험지(test)를 보고 고친 것이 아니라, 개발용(devel) 데이터에서만 정했다.

HTER은 이번 실험에서 완전히 제외했다. 따라서 저장되는 JSON에도 HTER을 넣지 않도록 구현했다.

### 4.4 효율성 측정 코드

파일: [src/analysis/compare_temporal_lightweight_efficiency.py](/home/saslab01/Desktop/replay-attack/src/analysis/compare_temporal_lightweight_efficiency.py)

이 파일은 각 모델 checkpoint를 다시 불러와서 효율성을 측정하고, 최종 비교표와 요약 파일을 만든다.

측정한 지표는 다음 4개다.

- Params: 모델 안에 있는 학습 가능한 숫자의 개수
- Size(MB): FP32 기준 모델 가중치 크기
- Latency(ms): 5-frame clip 하나를 처리하는 데 걸린 시간
- Peak GPU Memory(MB): 추론 중 GPU 메모리를 가장 많이 사용한 순간의 값

여기서 Latency는 실제 비디오 파일을 읽고 디코딩하는 전체 시간이 아니라 모델 forward pass 시간만 의미한다. Peak GPU Memory도 전체 시스템 메모리가 아니라, dummy clip 한 개를 모델에 넣었을 때의 GPU peak memory다.

비교표에는 기존 5-frame CNN-LSTM baseline도 포함했다. baseline checkpoint는 현재 프로젝트 내부 경로를 우선 사용했고, 필요할 때만 인접한 `../replay_pad` 경로를 fallback으로 복사하는 방식으로 처리했다.

## 5. 실제 실험 실행 흐름

실제 실험은 다음 순서로 진행했다.

1. 새 모델 코드가 제대로 작동하는지 pytest로 먼저 확인했다.
   - shape test
   - eval payload test
   - comparison column test
2. 기존 5-frame CNN-LSTM checkpoint를 현재 프로젝트 내부 경로로 복사했다.
3. CUDA 사용 가능 여부를 확인했고, NVIDIA GB10 GPU에서 학습했다.
4. MiniFASNet-LSTM을 10 epoch 학습하고 평가했다.
5. ShuffleNetV2-LSTM을 10 epoch 학습하고 평가했다.
6. EfficientNet-Lite-LSTM을 10 epoch 학습하고 평가했다.
7. MobileNetV4-LSTM을 10 epoch 학습하고 평가했다.
8. `compare_temporal_lightweight_efficiency.py`를 실행하여 최종 비교표와 요약 파일을 생성했다.
9. 새 결과 파일들에서 HTER이 제외됐는지 확인했다.

각 모델의 학습/평가 상태는 모두 완료되었다.

- MiniFASNet-LSTM: 학습/평가 완료
- ShuffleNetV2-LSTM: 학습/평가 완료
- EfficientNet-Lite-LSTM: 학습/평가 완료
- MobileNetV4-LSTM: 학습/평가 완료

이번 실험에서는 MobileNetV4나 EfficientNet-Lite가 구현 불가해서 보류되는 상황은 발생하지 않았다. 따라서 summary에 별도 실패 사유를 적을 모델은 없었다.

## 6. 실험 결과

아래 표는 [outputs/results/upgrade/temporal_lightweight_backbone_comparison.csv](/home/saslab01/Desktop/replay-attack/outputs/results/upgrade/temporal_lightweight_backbone_comparison.csv)의 수치를 정리한 것이다.

| Model | Backbone | Accuracy | APCER | BPCER | ACER | Params | Size(MB) | Latency(ms) | Peak GPU Mem(MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN-LSTM | MobileNetV3-Small | 0.995833 | 0.0 | 0.025 | 0.0125 | 1288738 | 4.916145 | 1.186081 | 26.587891 |
| MiniFASNet-LSTM | MiniFASNet | 1.0 | 0.0 | 0.0 | 0.0 | 139994 | 0.534035 | 0.638516 | 41.314941 |
| MobileNetV4-LSTM | MobileNetV4 | 1.0 | 0.0 | 0.0 | 0.0 | 490010 | 1.86924 | 0.688082 | 28.026855 |
| EfficientNet-Lite-LSTM | EfficientNet-Lite | 0.995833 | 0.005 | 0.0 | 0.0025 | 420266 | 1.603188 | 2.171534 | 70.354492 |
| ShuffleNetV2-LSTM | ShuffleNetV2 | 0.991667 | 0.0 | 0.05 | 0.025 | 1844710 | 7.03701 | 1.546491 | 35.675293 |

수치만 보면 다음과 같이 해석할 수 있다.

- 가장 낮은 ACER는 MiniFASNet-LSTM과 MobileNetV4-LSTM의 0.0이다.
- 가장 높은 Accuracy도 MiniFASNet-LSTM과 MobileNetV4-LSTM의 1.0이다.
- 가장 빠른 Latency는 MiniFASNet-LSTM의 0.638516 ms이다.
- 가장 낮은 Peak GPU Memory는 기존 CNN-LSTM의 26.587891 MB이다.

비교를 조금 더 비판적으로 보면 다음과 같다.

- MiniFASNet-LSTM은 성능 지표가 매우 좋고 Latency도 가장 빠르지만, Peak GPU Memory가 41.314941 MB로 baseline보다 높다.
- MobileNetV4-LSTM은 성능이 완벽하고 Latency도 매우 빠르며, Peak GPU Memory도 28.026855 MB로 MiniFASNet보다 훨씬 낮다. 따라서 성능과 효율의 균형 측면에서는 가장 설득력 있는 후보로 볼 수 있다.
- EfficientNet-Lite-LSTM은 ACER가 0.0025로 매우 낮지만, Latency가 2.171534 ms로 가장 느리고 Peak GPU Memory도 70.354492 MB로 가장 높다. 즉, 성능은 좋지만 단말기 친화성은 떨어진다.
- ShuffleNetV2-LSTM은 이름만 보면 매우 가벼울 것 같지만, 이번 구현에서는 Params가 1844710으로 오히려 가장 많고, ACER도 0.025로 baseline과 같거나 더 나쁜 편이다. 따라서 이 실험 조건에서는 좋은 선택이라고 보기 어렵다.
- 기존 CNN-LSTM은 Peak GPU Memory가 가장 낮아 메모리 측면에서 여전히 강점이 있지만, ACER는 0.0125로 MiniFASNet-LSTM과 MobileNetV4-LSTM보다 불리하다.

중요한 점은, Accuracy만 보면 MiniFASNet-LSTM과 MobileNetV4-LSTM이 가장 좋지만, 실제 단말기 적용을 생각하면 memory와 latency까지 함께 봐야 한다는 것이다. 성능이 좋아도 memory가 높으면 배포 부담이 커지고, memory가 낮아도 ACER가 높으면 탐지 성능 측면에서 한계가 있다.

## 7. 결과 해석: 성능 관점

각 지표의 의미는 다음과 같다.

- Accuracy: 전체 영상 중 맞힌 비율
- APCER: attack을 real로 잘못 판단한 비율. 보안성과 관련된다.
- BPCER: real을 attack으로 잘못 판단한 비율. 사용성과 관련된다.
- ACER: APCER와 BPCER를 함께 반영한 평균 오류율. 이번 실험의 핵심 성능 지표다.

이번 결과를 보면 다음과 같이 정리할 수 있다.

1. 기존 5-frame CNN-LSTM 대비 새 backbone들은 전반적으로 비슷하거나 더 좋은 ACER를 보였다.
2. 가장 좋은 성능은 MiniFASNet-LSTM과 MobileNetV4-LSTM이었다. 둘 다 test video-level에서 Accuracy 1.0, APCER 0.0, BPCER 0.0, ACER 0.0을 기록했다.
3. EfficientNet-Lite-LSTM은 APCER 0.005, ACER 0.0025로 꽤 좋았지만, 완전 무오류 수준까지는 가지 못했다.
4. ShuffleNetV2-LSTM은 BPCER 0.05로 real 영상을 spoof로 잘못 막는 오류가 남아 있어, accuracy와 ACER 모두에서 상대적으로 불리했다.
5. 기존 CNN-LSTM은 APCER 0.0, BPCER 0.025, ACER 0.0125로 baseline 역할은 잘 하지만, 최상위 성능은 아니었다.

정리하면, 성능만 놓고 보면 MiniFASNet-LSTM과 MobileNetV4-LSTM이 가장 강하다. 다만 MiniFASNet-LSTM은 메모리가 더 크고, MobileNetV4-LSTM은 성능이 같으면서 memory가 훨씬 낮아 더 균형적이다.

## 8. 결과 해석: 효율성 관점

효율성 지표는 모델을 실제 장치에 올릴 때 특히 중요하다.

- Params는 모델이 얼마나 큰지를 직관적으로 보여준다.
- Size(MB)는 가중치 파일이 대략 얼마나 큰지 보여준다.
- Latency(ms)는 한 번 예측하는 데 얼마나 오래 걸리는지 보여준다.
- Peak GPU Memory(MB)는 추론 중 메모리 여유가 얼마나 필요한지 보여준다.

이번 결과를 보면 다음과 같다.

1. 가장 빠른 모델은 MiniFASNet-LSTM이다.
2. 가장 작은 모델 크기는 MiniFASNet-LSTM이다.
3. 가장 낮은 GPU memory는 기존 CNN-LSTM이다.
4. 가장 느린 모델은 EfficientNet-Lite-LSTM이다.
5. 가장 memory를 많이 쓴 모델은 EfficientNet-Lite-LSTM이다.

단말기 적용 관점에서 보면, MiniFASNet-LSTM은 속도는 좋지만 memory 부담이 있고, EfficientNet-Lite-LSTM은 성능은 좋지만 속도와 memory가 부담된다. MobileNetV4-LSTM은 성능이 완벽하면서도 latency와 memory가 함께 균형적이어서, 실제 적용 후보로 설명하기 가장 무난하다. 기존 CNN-LSTM은 memory가 가장 낮아서 여전히 매력적이지만, 성능이 완전히 최고는 아니다.

## 9. 비판적 검토와 한계

교수님 랩미팅에서 같이 검토해야 할 한계도 분명하다.

1. 모든 모델은 Replay-Attack 하나의 데이터셋에서만 검증했다.
2. 모든 모델은 random initialization이므로, 사전학습 기반 최신 모델의 최대 성능을 비교한 것은 아니다.
3. Latency와 memory는 실제 비디오 decoding/end-to-end 서비스 시간이 아니라 model forward pass 기준이다.
4. 효율성 측정은 batch size 1, 5-frame dummy input 기준이다.
5. 실제 배포에서는 CPU latency, RAM, 전력 소모, 실제 카메라 입력 처리 시간 등을 추가로 봐야 한다.
6. 단일 seed 결과이므로, 더 강한 일반화를 주장하려면 여러 seed 반복 실험이 필요하다.
7. test 결과를 보고 hyperparameter를 바꾸지 않았기 때문에 공정성은 유지했지만, 모델별 최적 튜닝까지 수행한 것은 아니다.
8. 이번 구현은 외부 pretrained checkpoint를 쓰지 않았기 때문에, 산업용 벤치마크와 1:1로 동일한 결과라고 보기는 어렵다.

즉, 이번 실험은 “절대 최고 성능”을 주장하려는 것이 아니라, 동일한 5-frame temporal setting에서 backbone 선택이 성능과 효율성에 어떤 영향을 주는지 보여주는 비교 실험으로 보는 것이 맞다.

## 10. 발표/랩미팅용 핵심 요약 멘트

“기존 ASK 논문에서 사용한 5-frame CNN-LSTM 구조는 유지하되, backbone만 MobileNetV3-Small, MiniFASNet, MobileNetV4, EfficientNet-Lite, ShuffleNetV2로 바꿔서 성능과 효율성을 비교했습니다. 모든 모델은 5-frame clip 입력과 LSTM temporal head를 동일하게 사용했고, pretrained 없이 random initialization으로 Replay-Attack에서 학습했습니다. 평가는 clip-level이 아니라 video-level로 했고, threshold는 devel split에서만 고른 뒤 test에는 고정 적용했습니다. HTER은 이번 실험에서 제외했습니다. 결과적으로 MiniFASNet-LSTM과 MobileNetV4-LSTM이 가장 낮은 ACER를 보였고, MiniFASNet-LSTM이 가장 빨랐지만 memory는 더 컸습니다. 반면 MobileNetV4-LSTM은 성능이 같으면서 memory가 더 낮아, 발표에서는 단순 최고 정확도보다 성능-효율 trade-off 관점에서 가장 설득력 있는 후속 연구로 설명할 수 있습니다.” 

## 11. 산출물 목록

- [docs/superpowers/specs/2026-05-04-temporal-lightweight-backbone-comparison-design.md](/home/saslab01/Desktop/replay-attack/docs/superpowers/specs/2026-05-04-temporal-lightweight-backbone-comparison-design.md)
- [docs/superpowers/plans/2026-05-04-temporal-lightweight-backbone-comparison.md](/home/saslab01/Desktop/replay-attack/docs/superpowers/plans/2026-05-04-temporal-lightweight-backbone-comparison.md)
- [src/models/temporal_lightweight_lstm.py](/home/saslab01/Desktop/replay-attack/src/models/temporal_lightweight_lstm.py)
- [src/engine/train_temporal_lightweight_lstm.py](/home/saslab01/Desktop/replay-attack/src/engine/train_temporal_lightweight_lstm.py)
- [src/engine/evaluate_temporal_lightweight_lstm.py](/home/saslab01/Desktop/replay-attack/src/engine/evaluate_temporal_lightweight_lstm.py)
- [src/analysis/compare_temporal_lightweight_efficiency.py](/home/saslab01/Desktop/replay-attack/src/analysis/compare_temporal_lightweight_efficiency.py)
- [outputs/checkpoints/upgrade/](/home/saslab01/Desktop/replay-attack/outputs/checkpoints/upgrade)
- [outputs/results/upgrade/](/home/saslab01/Desktop/replay-attack/outputs/results/upgrade)
- [outputs/results/upgrade/temporal_lightweight_backbone_comparison.csv](/home/saslab01/Desktop/replay-attack/outputs/results/upgrade/temporal_lightweight_backbone_comparison.csv)
- [outputs/results/upgrade/temporal_lightweight_backbone_comparison.md](/home/saslab01/Desktop/replay-attack/outputs/results/upgrade/temporal_lightweight_backbone_comparison.md)
- [outputs/results/upgrade/temporal_lightweight_backbone_summary.md](/home/saslab01/Desktop/replay-attack/outputs/results/upgrade/temporal_lightweight_backbone_summary.md)
- [outputs/results/upgrade/temporal_lightweight_backbone_experiment_report.md](/home/saslab01/Desktop/replay-attack/outputs/results/upgrade/temporal_lightweight_backbone_experiment_report.md)

## 12. 최종 정리

이번 후속 실험의 핵심은 “최고 정확도 하나”를 고르는 것이 아니라, 같은 5-frame temporal order 조건에서 backbone 선택이 성능과 효율에 어떤 차이를 만드는지 보여주는 데 있다. 실험 결과만 놓고 보면, MiniFASNet-LSTM과 MobileNetV4-LSTM이 가장 강한 성능을 보였고, 그중 MobileNetV4-LSTM은 성능과 효율을 함께 봤을 때 가장 균형적인 후보로 해석할 수 있다. 반대로 EfficientNet-Lite-LSTM은 성능은 좋지만 latency와 memory가 부담되고, ShuffleNetV2-LSTM은 이번 구현과 조건에서는 기대만큼 좋은 trade-off를 보이지 않았다.

따라서 발표에서는 이 결과를 “기존 5-frame CNN-LSTM을 대체한다”는 식으로 과장하기보다, “같은 temporal 조건을 유지하면서도 backbone 선택만으로 성능-효율 trade-off를 더 잘 조정할 수 있음을 보였다”는 후속 연구 근거로 제시하는 것이 가장 적절하다.

