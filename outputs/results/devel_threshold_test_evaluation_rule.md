# Devel Threshold and Test Evaluation Rule

- Threshold selection: each evaluation script computes spoof scores on the devel split, aggregates them at video level, and searches thresholds from 0.0 to 1.0 with step 0.001 to minimize devel ACER.
- Test evaluation: the selected devel threshold is fixed and applied to the test video-level scores. No threshold is re-selected on the test split.
- Video-level aggregation: 1-frame predictions are grouped by `video_id` and averaged. The 5-frame and 10-frame average settings first average frame scores within each sequence, then group sequence scores by `video_id` and average again. CNN-LSTM first predicts clip scores, then groups clip scores by `video_id` and averages.
- Code check: the current code matches the intended rule for the four requested models. Result JSON files also record `threshold_selected_on: devel`.
