# 5-frame CNN-LSTM Student 오류 분석 논문용 정리

## 1. 이 분석을 수행한 목적

본 연구의 핵심 목표는 Replay-Attack 데이터셋에서 경량 MobileNetV3-Small backbone 기반 face anti-spoofing 모델에 temporal modeling을 적용했을 때, 성능 향상과 계산 비용 사이의 관계를 정량적으로 분석하는 것이다. 앞선 실험에서 10-frame CNN-LSTM은 1-frame 및 frame-average 방식보다 낮은 ACER를 보였지만, peak GPU memory가 증가했다. 이에 따라 후속 실험에서는 temporal length를 10 frame에서 5 frame으로 줄인 5-frame CNN-LSTM student를 구성하여, temporal modeling의 이점을 유지하면서 memory 및 latency 비용을 낮출 수 있는지 확인하였다.

5-frame CNN-LSTM student는 test video-level 기준 Accuracy 0.995833, APCER 0.0000, BPCER 0.0250, ACER 0.0125를 달성했다. 이는 5-frame average 방식보다 훨씬 낮은 ACER를 보이면서도 10-frame CNN-LSTM보다 낮은 peak GPU memory와 latency를 보이는 결과였다. 그러나 ACER가 0이 아니며, 최종 오류가 어디에서 발생했는지 확인해야 모델의 한계와 개선 방향을 논문에서 설득력 있게 설명할 수 있다.

따라서 본 오류 분석의 목적은 단순히 최종 성능 수치를 보고하는 것이 아니라, 5-frame CNN-LSTM student가 어떤 조건에서 불안정해지는지, 오류가 attack sample을 놓치는 문제인지 genuine sample을 attack으로 오인하는 문제인지, 그리고 이후 성능 개선을 위해 어떤 feature 또는 calibration 방향이 필요한지를 확인하는 것이다.

## 2. 실험 구성

분석 대상 모델은 5-frame CNN-LSTM student이다. 입력은 한 video에서 생성된 5-frame clip이며, 각 frame은 MobileNetV3-Small 기반 CNN feature extractor를 통과한 뒤 LSTM에 순차적으로 입력된다. LSTM의 마지막 hidden representation을 이용하여 binary classification을 수행한다. 이 모델은 5-frame clip 단위로 spoof score를 출력하고, video-level 평가에서는 같은 video에 속한 clip score들을 평균하여 하나의 video score를 만든다.

평가는 Replay-Attack의 devel/test split을 사용했다. Threshold는 test set에서 조정하지 않고 devel set에서만 선택하였다. Devel video-level score에 대해 ACER가 가장 낮은 threshold를 탐색했고, 선택된 threshold는 0.103이었다. 이후 test set에서는 이 threshold를 고정한 상태로 video-level score에만 적용하여 Accuracy, APCER, BPCER, ACER를 계산했다.

실험 설정은 다음과 같다.

| 항목 | 설정 |
|---|---|
| Model | 5-frame CNN-LSTM student |
| Dataset | Replay-Attack |
| Input unit | 5-frame clip |
| Video aggregation | Mean of clip-level spoof scores |
| Threshold source | Devel split |
| Threshold | 0.103 |
| Test rule | Fixed threshold applied to test video-level scores |
| Positive class | Attack |
| Negative class | Real |

최종 test video-level 결과는 다음과 같다.

| Accuracy | APCER | BPCER | ACER | TP | TN | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.995833 | 0.0000 | 0.0250 | 0.0125 | 400 | 78 | 2 | 0 |

이 결과에서 attack video를 real로 잘못 판단한 FN은 없었다. 모든 오류는 real video를 attack으로 잘못 판단한 FP였다. 따라서 5-frame CNN-LSTM student의 test set 오류는 APCER 문제가 아니라 BPCER 문제로 해석된다.

## 3. 오류 분석 절차

오류 분석은 네 단계로 진행했다.

첫째, test video-level prediction 파일에서 `correct=0`인 video를 추출했다. 이 단계에서는 전체 test video 중 어떤 sample이 최종적으로 오분류되었는지 확인했다.

둘째, 각 오분류 video에 대해 clip-level prediction을 다시 연결했다. Video-level score는 clip score의 평균이므로, 최종 오분류가 몇 개의 outlier clip 때문에 발생했는지, 아니면 video 전체 clip이 일관되게 잘못 분류되었는지를 확인하기 위해 clip score의 min, max, standard deviation, threshold 초과 비율을 계산했다.

셋째, devel/test score 분포를 비교했다. Devel에서 선택된 threshold가 test에 적절히 적용되었는지 확인하고, devel real score와 test real score의 범위를 비교하여 threshold selection 문제인지 generalization 문제인지 구분했다.

넷째, 오분류 sample의 visual/low-level 특성을 확인했다. 같은 client의 controlled real video와 adverse real video를 비교하기 위해 contact sheet를 생성했고, brightness, contrast, saturation, sharpness, temporal frame difference 등 간단한 영상 통계를 계산했다. 이 분석은 모델 내부 attention이나 feature attribution을 직접 계산한 것은 아니지만, 오류가 특정 조명/배경/화질 조건과 함께 나타나는지 확인하기 위한 diagnostic analysis로 사용했다.

분석에 사용한 주요 산출물은 다음과 같다.

| 파일 | 용도 |
|---|---|
| `outputs/predictions/cnn_lstm_clip5_test_video_predictions_annotated.csv` | test video-level prediction 및 error type 확인 |
| `outputs/predictions/cnn_lstm_clip5_test_clip_predictions.csv` | clip-level score trace 확인 |
| `outputs/analysis/devel_errors/cnn_lstm_clip5_test_misclassified.csv` | 최종 오분류 video 목록 |
| `outputs/analysis/clip5_error_analysis/clip5_video_score_stability.csv` | 전체 test video의 clip score 안정성 요약 |
| `outputs/analysis/clip5_error_analysis/clip5_fp_clip_score_trace.csv` | FP video의 clip별 score 변화 |
| `outputs/analysis/clip5_error_analysis/client014_real_contact_sheet.jpg` | client014 real video visual 비교 |
| `outputs/analysis/clip5_error_analysis/clip5_focus_video_diagnostics.csv` | 오분류 및 위험 sample의 low-level diagnostic feature |

## 4. 오류가 발생한 sample

Test set에서 발생한 2개의 오류는 모두 같은 subject인 `client014`의 real/adverse video였다.

| Video | GT | Pred | Score | Threshold | Error |
|---|---:|---:|---:|---:|---|
| `test__real__real__client014_session01_webcam_authenticate_adverse_1` | real | attack | 0.852429 | 0.103 | FP |
| `test__real__real__client014_session01_webcam_authenticate_adverse_2` | real | attack | 0.974687 | 0.103 | FP |

같은 client의 controlled real video는 모두 정상적으로 real로 분류되었다.

| Video condition | Score | Prediction | Result |
|---|---:|---|---|
| `client014 adverse_1` | 0.852429 | attack | FP |
| `client014 adverse_2` | 0.974687 | attack | FP |
| `client014 controlled_1` | 0.000476 | real | Correct |
| `client014 controlled_2` | 0.000483 | real | Correct |

이 결과는 모델이 특정 identity 자체를 항상 attack으로 보는 것이 아니라, 같은 identity라도 adverse condition에서만 크게 흔들린다는 점을 보여준다. 따라서 오류 원인은 subject identity 단독이라기보다 adverse illumination, background, camera condition, face appearance 변화가 결합된 조건에서 발생한 것으로 해석할 수 있다.

## 5. Clip-level score 분석

두 FP video는 단일 clip outlier 때문에 video 평균 score가 상승한 사례가 아니었다. 각 video는 16개의 5-frame clip으로 구성되었고, 두 video 모두 16개 clip 전부가 devel threshold 0.103을 초과했다.

| Video | Video score | Clip min | Clip max | Clip std | Clips over threshold |
|---|---:|---:|---:|---:|---:|
| `client014 adverse_1` | 0.852429 | 0.527562 | 0.988363 | 0.162401 | 16 / 16 |
| `client014 adverse_2` | 0.974687 | 0.939318 | 0.992302 | 0.017864 | 16 / 16 |

`client014 adverse_1`은 clip score의 표준편차가 0.162401로 비교적 크며, video 중간 구간에서 score가 0.527562까지 내려간다. 그러나 최저 clip score도 threshold 0.103보다 충분히 높기 때문에 최종 video-level prediction은 여전히 attack이 된다. `client014 adverse_2`는 clip score의 표준편차가 0.017864로 작고, 모든 clip이 0.939 이상이다. 즉, 이 video는 시간적으로도 일관되게 attack처럼 분류된다.

이 결과는 aggregation 방식만 바꾸는 단순한 후처리로는 오류를 해결하기 어렵다는 것을 의미한다. 예를 들어 mean aggregation을 median aggregation으로 바꾸더라도 `adverse_2`는 거의 모든 clip이 0.95 이상이기 때문에 여전히 attack으로 분류될 가능성이 높다. 따라서 개선은 clip score aggregation보다는 feature representation, real/adverse robustness, score calibration 쪽에서 접근하는 것이 더 타당하다.

## 6. Devel threshold와 test generalization 분석

Devel에서 선택된 threshold는 0.103이다. Devel split에서는 real score와 attack score가 완전히 분리된다.

| Split | Max real score | Min attack score | Threshold |
|---|---:|---:|---:|
| Devel | 0.102234 | 0.801749 | 0.103 |
| Test | 0.974687 | 0.757971 | 0.103 |

Devel 기준으로 보면 threshold 0.103은 모든 real video를 threshold 아래에 두고, 모든 attack video를 threshold 위에 두는 값이다. 따라서 코드 구현상 threshold selection은 정상적으로 작동한다. 그러나 devel의 max real score가 0.102234이고 threshold가 0.103이므로, devel real sample에 대한 safety margin은 0.000766에 불과하다. 이는 devel set에서 boundary 근처의 real sample이 존재했음을 의미한다.

Test set에서는 real video 중 두 개가 0.852429와 0.974687까지 상승했다. 이는 devel에서 관찰된 real score 범위를 크게 벗어난다. 따라서 오류의 핵심은 test threshold 적용 방식이 아니라, devel에서 관찰되지 않았거나 충분히 반영되지 않은 adverse genuine condition에 대한 generalization failure이다.

Test threshold를 사후적으로 바꾸면 다음과 같은 trade-off가 나타난다. 이 분석은 final evaluation을 바꾸기 위한 것이 아니라, score distribution의 성격을 이해하기 위한 diagnostic analysis이다.

| Threshold | FP | FN | APCER | BPCER | ACER |
|---:|---:|---:|---:|---:|---:|
| 0.103 | 2 | 0 | 0.0000 | 0.0250 | 0.0125 |
| 0.900 | 1 | 2 | 0.0050 | 0.0125 | 0.0088 |
| 0.975 | 0 | 5 | 0.0125 | 0.0000 | 0.0063 |

Threshold를 높이면 FP는 줄어들지만 일부 attack video가 FN으로 바뀐다. 따라서 단순 threshold 조정은 BPCER와 APCER 사이의 trade-off를 이동시킬 뿐이며, 근본적으로 adverse real sample의 score를 낮추는 feature 또는 calibration 개선이 필요하다.

## 7. Visual 및 low-level diagnostic 결과

오분류 sample을 같은 client의 controlled real sample과 비교하기 위해 contact sheet를 생성했다. Contact sheet에서는 `client014`의 adverse real video와 controlled real video를 같은 row structure로 배치하여 시각적으로 비교했다.

Contact sheet 파일:

`outputs/analysis/clip5_error_analysis/client014_real_contact_sheet.jpg`

시각적으로 FP가 발생한 adverse real video는 controlled real video와 배경 및 조명 조건이 다르다. 특히 adverse video는 창문/배경 구조와 강한 조명 변화가 포함되어 있으며, controlled video는 배경이 비교적 균일하고 얼굴 appearance가 안정적이다.

Low-level diagnostic feature에서도 차이가 확인되었다.

| Video group | Brightness | Contrast | Saturation | Sharpness |
|---|---:|---:|---:|---:|
| `client014 adverse real FP` | 0.549-0.556 | 0.300-0.304 | 0.497-0.498 | 0.00336-0.00340 |
| `client014 controlled real correct` | 0.542-0.548 | 0.225 | 0.755-0.759 | 0.00114-0.00119 |

이 수치는 모델이 실제로 해당 feature를 사용했다는 직접적인 증거는 아니다. 그러나 오류가 발생한 video가 controlled real video보다 높은 contrast와 sharpness를 보이고, saturation 통계도 크게 다르다는 점은 모델이 spoof-specific temporal cue뿐 아니라 조명/배경/화질과 관련된 appearance cue에 민감할 가능성을 시사한다.

## 8. 모델이 취약해지는 조건

본 분석에서 확인한 5-frame CNN-LSTM student의 취약 조건은 다음과 같다.

첫째, genuine adverse condition에 취약하다. Test error는 모두 real/adverse video에서 발생했으며, attack video에 대한 miss는 없었다. 따라서 이 모델의 현재 약점은 attack detection sensitivity 부족이 아니라 genuine sample을 attack으로 과도하게 판단하는 specificity 부족이다.

둘째, 특정 client-environment 조합에서 오류가 집중된다. 오류는 모두 `client014`의 adverse real video에서 발생했다. 같은 client의 controlled real video는 near-zero score로 정확히 분류되므로, identity 자체보다는 environment 또는 acquisition condition이 핵심 요인으로 보인다.

셋째, 일부 video에서는 clip score가 시간적으로 불안정하다. `client014 adverse_1`은 clip score가 0.527562부터 0.988363까지 변하며, clip score std가 0.162401이다. 또한 일부 correctly classified attack video에서도 high variance가 관찰된다. 이는 5-frame clip 기반 temporal representation이 짧은 구간의 appearance 변화나 motion condition에 민감하게 반응할 수 있음을 보여준다.

넷째, devel과 test 사이에 calibration gap이 존재한다. Devel real score의 최대값은 0.102234였지만, test real score는 0.974687까지 상승했다. 이 gap은 devel threshold가 구현상 올바르더라도, test의 특정 genuine condition을 충분히 커버하지 못할 수 있음을 보여준다.

## 9. 결과의 의미

5-frame CNN-LSTM student는 전체 성능 관점에서는 매우 높은 accuracy와 낮은 ACER를 달성했다. 또한 FN이 없다는 점에서 attack detection 측면은 안정적이다. 그러나 오류 분석 결과, 모델은 일부 adverse genuine video에 대해 매우 높은 attack confidence를 출력한다. 이는 face anti-spoofing에서 중요한 문제다. 실제 응용에서는 genuine user를 attack으로 거부하는 BPCER 증가가 usability를 저하시킬 수 있기 때문이다.

특히 두 FP video의 모든 clip이 threshold를 초과했다는 점은 중요한 해석 포인트다. 만약 일부 clip만 잘못 분류되었다면 video-level robust aggregation으로 해결할 수 있다. 하지만 본 실험에서는 video 전체가 attack-like representation으로 매핑되었기 때문에, 더 근본적인 feature representation 또는 calibration 개선이 필요하다.

따라서 5-frame CNN-LSTM student의 다음 개선 방향은 temporal length를 무작정 늘리는 것이 아니라, 짧은 temporal model의 효율성을 유지하면서 adverse real condition에 대한 overconfidence를 줄이는 것이다. 이는 본 논문의 핵심 메시지인 "temporal modeling의 성능 이득과 비용"에 자연스럽게 연결된다. 5-frame student는 cost-effective한 temporal model이지만, 최종 오류는 adverse genuine robustness와 score calibration에서 발생한다.

## 10. 개선 feature 및 후속 실험 방향

본 분석을 바탕으로 다음 개선 방향을 제안할 수 있다.

### 10.1 Illumination/quality-aware calibration

Brightness, contrast, saturation, sharpness, temporal frame difference와 같은 lightweight quality feature를 이용하여 score calibration을 수행하는 방향이다. 이 feature들은 모델의 주 classifier를 대체하기 위한 것이 아니라, adverse real condition에서 지나치게 높은 attack score가 발생하는지를 보정하기 위한 auxiliary diagnostic 또는 calibration feature로 사용할 수 있다.

논문에서의 해석은 다음과 같이 정리할 수 있다.

> The error analysis indicates that the remaining false positives are concentrated in genuine videos captured under adverse illumination. Therefore, a lightweight quality-aware calibration module may be useful for suppressing overconfident attack scores on difficult genuine samples without increasing the temporal length.

### 10.2 Real/adverse-aware augmentation or sampling

현재 오류는 attack sample 부족이 아니라 real/adverse condition의 generalization 문제로 나타났다. 따라서 augmentation을 적용한다면 attack augmentation보다 genuine/adverse robustness를 높이는 방향이 더 직접적이다. 예를 들어 brightness/contrast perturbation, background/illumination variation, face crop normalization을 real sample에도 충분히 적용할 수 있다.

주의할 점은 새로운 augmentation 실험은 현재 논문 본 실험 범위를 확장할 수 있으므로, 본문에서는 future work 또는 follow-up experiment로 다루는 것이 안전하다.

### 10.3 Teacher-guided 5-frame student calibration

10-frame CNN-LSTM teacher는 더 긴 temporal evidence를 사용하지만 memory cost가 크다. 5-frame student는 효율성이 좋지만 adverse real sample에서 overconfidence가 발생한다. 따라서 10-frame teacher의 soft score를 활용해 5-frame student를 calibration하거나 distillation하는 방향이 자연스럽다.

이 방향은 본 논문의 효율성 메시지와도 잘 맞는다. 즉, temporal length를 줄여 memory/latency를 낮추되, teacher-guided calibration으로 짧은 student의 score 안정성을 보완하는 전략이다.

### 10.4 Robust video-level aggregation

Mean aggregation 외에 median, trimmed mean, uncertainty-aware aggregation을 비교할 수 있다. 다만 본 분석에서는 두 FP video 모두 모든 clip이 threshold를 초과했으므로, aggregation 변경만으로는 충분하지 않을 가능성이 높다. 따라서 robust aggregation은 단독 해결책이라기보다 score stability analysis와 함께 보조적으로 고려하는 것이 적절하다.

## 11. 논문에 바로 사용할 수 있는 문장

아래 문장들은 논문 본문 또는 discussion section에 바로 옮겨 쓸 수 있는 형태로 정리한 것이다.

> To better understand the failure modes of the 5-frame CNN-LSTM student, we performed a video- and clip-level error analysis using the fixed threshold selected on the devel split.

> The model achieved 99.58% video-level accuracy with an ACER of 1.25% on the test split. All remaining errors were false positives, indicating that the primary weakness of the 5-frame student is not missed attacks but overconfident rejection of genuine videos.

> The two false-positive samples were both genuine videos from the same subject under adverse illumination. In contrast, the controlled genuine videos of the same subject were correctly classified with near-zero spoof scores, suggesting that the failure is associated with acquisition condition rather than identity alone.

> Clip-level analysis showed that all 16 clips in both false-positive videos exceeded the devel-selected threshold. Therefore, the errors were not caused by isolated outlier clips, and simple video-level score aggregation is unlikely to fully resolve this failure mode.

> The devel split showed perfect separation between genuine and attack videos at the selected threshold, whereas the test split contained adverse genuine videos with spoof scores far outside the devel genuine range. This indicates a calibration/generalization gap under adverse genuine conditions.

> These observations suggest that future improvements should focus on reducing overconfident attack scores for adverse genuine videos, for example through quality-aware calibration, real-adverse augmentation, or teacher-guided calibration of the 5-frame student.

## 12. 본문 배치 제안

이 분석은 논문의 main result table 직후 또는 discussion section에 배치하는 것이 적절하다. 추천 흐름은 다음과 같다.

1. 먼저 1-frame, 5-frame avg, 10-frame avg, 10-frame CNN-LSTM, 5-frame CNN-LSTM student의 성능 및 효율성 결과를 제시한다.
2. 그 다음 5-frame student가 memory/latency 측면에서 효율적인 대안임을 설명한다.
3. 이어서 remaining error analysis를 제시하여, 5-frame student의 오류가 어디에서 발생하는지 설명한다.
4. 마지막으로 이 오류가 future work의 방향, 즉 adverse genuine robustness 및 score calibration 개선으로 이어진다고 정리한다.

이렇게 배치하면 단순히 "성능이 좋다"에서 끝나지 않고, 모델의 한계와 다음 개선 방향까지 논리적으로 연결할 수 있다.

