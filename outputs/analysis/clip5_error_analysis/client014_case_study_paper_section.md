# Client014 False Positive Case Study

## Paper-ready subsection draft

### Error case analysis on genuine adverse videos

To further examine the remaining errors of the 5-frame CNN-LSTM student, we conducted a case-level analysis of the misclassified test videos. The model produced only two video-level errors on the Replay-Attack test set, and both errors occurred on genuine videos from the same subject, `client014`, under the adverse condition. No attack video was misclassified as genuine. Therefore, the remaining failure mode of the 5-frame student is concentrated on false rejection of genuine samples rather than missed attack detection.

The two misclassified videos were `test__real__real__client014_session01_webcam_authenticate_adverse_1` and `test__real__real__client014_session01_webcam_authenticate_adverse_2`. Although both videos are genuine, the model assigned high attack scores of 0.852429 and 0.974687, respectively, which are far above the devel-selected threshold of 0.103. In contrast, the two controlled genuine videos from the same subject were correctly classified with near-zero scores of 0.000476 and 0.000483. This indicates that the error is not caused by the subject identity alone, but is strongly associated with the adverse acquisition condition.

Clip-level analysis showed that the false positives were not caused by isolated outlier clips. Each false-positive video consisted of 16 clips, and all 16 clips exceeded the decision threshold. For `client014 adverse_1`, the clip scores ranged from 0.527562 to 0.988363, with a standard deviation of 0.162401. For `client014 adverse_2`, the scores ranged from 0.939318 to 0.992302, with a standard deviation of 0.017864. Thus, even the lowest clip score in both videos was substantially higher than the threshold. This suggests that simple video-level aggregation changes, such as replacing mean aggregation with median aggregation, are unlikely to fully correct this failure mode.

Visual inspection further suggests that the false-positive videos differ from the controlled genuine videos in illumination and background conditions. The adverse videos include a bright window/background structure and stronger appearance variation, whereas the controlled videos show a more uniform background and stable facial appearance. Consistent with this observation, low-level diagnostics showed that the false-positive adverse videos had higher contrast and sharpness than the controlled genuine videos from the same subject. Although these statistics do not directly prove which cues the network used, they indicate that the student model may be sensitive to appearance or acquisition-condition cues that do not generalize reliably to adverse genuine samples.

This case study reveals a calibration and robustness limitation of the 5-frame CNN-LSTM student. The devel threshold is valid under the implemented evaluation protocol, but the test set contains genuine adverse samples whose scores lie far outside the genuine score range observed in devel. Therefore, the most relevant improvement direction is not simply increasing attack sensitivity, but reducing overconfident attack predictions for difficult genuine videos. Possible follow-up directions include quality-aware score calibration, real-adverse augmentation, and teacher-guided calibration from the 10-frame CNN-LSTM model.

## Shorter version for main text

The remaining errors of the 5-frame CNN-LSTM student were concentrated in two genuine adverse videos from `client014`. Both videos were falsely classified as attacks with high video-level scores of 0.852429 and 0.974687, whereas the controlled genuine videos of the same subject were correctly classified with near-zero scores. Clip-level analysis showed that all 16 clips in each false-positive video exceeded the devel-selected threshold, indicating that the errors were not caused by isolated outlier clips. This suggests that the 5-frame student is vulnerable to adverse genuine acquisition conditions and may rely on appearance or illumination cues that are not fully robust across environments.

## Table for paper

| Subject | Condition | Video | Ground truth | Prediction | Video score | Threshold | Result |
|---|---|---|---|---|---:|---:|---|
| client014 | adverse | `adverse_1` | real | attack | 0.852429 | 0.103 | FP |
| client014 | adverse | `adverse_2` | real | attack | 0.974687 | 0.103 | FP |
| client014 | controlled | `controlled_1` | real | real | 0.000476 | 0.103 | Correct |
| client014 | controlled | `controlled_2` | real | real | 0.000483 | 0.103 | Correct |

## Clip-level table for paper

| Video | Number of clips | Clip min | Clip max | Clip std | Clips above threshold |
|---|---:|---:|---:|---:|---:|
| `client014 adverse_1` | 16 | 0.527562 | 0.988363 | 0.162401 | 16/16 |
| `client014 adverse_2` | 16 | 0.939318 | 0.992302 | 0.017864 | 16/16 |

## Figure caption draft

Figure files:

- `outputs/analysis/clip5_error_analysis/client014_case_study_figure.png`
- `outputs/analysis/clip5_error_analysis/client014_case_study_figure.pdf`

**Figure X. False-positive examples from `client014` under adverse acquisition conditions.** The 5-frame CNN-LSTM student falsely classified two genuine adverse videos as attacks, while the controlled genuine videos from the same subject were correctly classified with near-zero spoof scores. The adverse videos contain stronger illumination/background variation, suggesting that the remaining errors are associated with acquisition-condition sensitivity rather than subject identity alone.

Before including this figure in the paper, confirm that the Replay-Attack dataset license permits publication of identifiable subject frames in qualitative examples. If needed, use a blurred or cropped version for privacy-preserving presentation.

## Interpretation sentence options

1. The two remaining errors are false positives from genuine adverse videos, indicating that the 5-frame student primarily suffers from genuine-sample over-rejection rather than missed attack detection.

2. Since all clips in both false-positive videos exceed the devel-selected threshold, the failure is not due to isolated clip-level outliers but to consistently high attack confidence across the entire video.

3. The controlled videos of the same subject are classified correctly with near-zero scores, suggesting that the failure is related to adverse acquisition conditions rather than identity alone.

4. This case motivates future work on quality-aware calibration or real-adverse robustness enhancement for lightweight temporal anti-spoofing models.
