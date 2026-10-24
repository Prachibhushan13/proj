# Model and inference notes

## Active checkpoint and collapsed ONNX artifact

`backend/artifacts/models/cnn_lstm_best.onnx` is present, with input shape
`(batch, 20, 224, 224, 3)` and one sigmoid output. The training label convention
is `0 = real`, `1 = deepfake`, so the output is the fake-class score. However,
the ONNX artifact returns nearly the same score on unrelated and user-identified
real/AI videos. Its source `.keras` checkpoint was absent from this checkout.
The ONNX file is a collapsed checkpoint and should not be used. The local app
and Docker environment now select the newer
`backend/artifacts/models/cnn_lstm_celebdf_balanced100.keras` checkpoint by
default. Its output changes across videos, unlike the ONNX file, but measured
performance remains near chance.

## Training preprocessing and fixes

The CNN-LSTM training path now uses the application's sequence preparation:
decode the original video, sample with `FRAME_STRIDE`, detect/crop RGB faces,
resize to 224×224, scale pixels to [0, 1], keep up to 20 ordered face crops,
and repeat the last crop to fill short sequences. Videos with no decodable
frames or no detectable faces fail with an explicit error rather than becoming
all-zero training examples.

Keras EfficientNetB0 includes its own 1/255 rescaling and expects [0, 255]
pixels. The model graph now restores that range internally, preserving the
application's [0, 1] input contract. The training generator includes the final
partial batch, shuffles only the training split, and applies inverse-frequency
class weights because Celeb-DF has substantially more manipulated than real
videos. The final sigmoid remains float32 under mixed precision.

## Dataset and evaluation

The repo's Celeb-DF manifest uses video-level `train`, `val`, and official
`test` splits with labels `0 = real`, `1 = deepfake`. The full Kaggle Celeb-DF
v2 archive is now downloaded and extracted; the regenerated manifest contains
6,529 videos and the official 518-video test split. A pilot checkpoint was
trained before the archive completed, from the intact videos then available:
40 train, 40 validation, and 40 official test videos (20 of each class per
split), for 3 epochs with the EfficientNet backbone frozen. The download had
912 of 6,529 manifest videos available at that time. The pilot's official
held-out test results were:

- Accuracy: 47.5%
- Precision (deepfake class): 48.4%
- Recall (deepfake class): 75.0%
- F1 (deepfake class): 58.8%
- ROC-AUC: 0.53
- Confusion matrix, rows=true `[real, deepfake]`, columns=predicted
  `[real, deepfake]`: `[[4, 16], [5, 15]]`

These results are near chance. The newer Keras checkpoint was evaluated on a
deterministic balanced sample of 40 videos from the official test split, with
the same preprocessing and threshold used by application inference. Results:

- Accuracy: 50.0%
- Fake precision: 50.0%
- Fake recall: 45.0%
- Fake F1: 47.4%
- ROC-AUC: 0.5425
- Confusion matrix, rows=true `[real, deepfake]`, columns=predicted
  `[real, deepfake]`: `[[11, 9], [11, 9]]`

The per-video sample, scores, checkpoint hash, and manifest hash are in
`backend/artifacts/evaluation/active_keras_test_sample.json`. The evaluator is
`scripts/evaluate_active_keras_model.py`. This 40-video sample is not the full
official test split; these results do not establish reliability.

The complete 6,529-video dataset is present. To train on it, regenerate the
manifest and run:

```bash
python scripts/prepare_celebdf.py
python scripts/train_model.py --manifest datasets/celebdf_manifest.csv \
  --dataset-root datasets/Celeb-DF-v2 --epochs 20 --batch-size 2
```

Training selects the checkpoint using validation loss. The held-out test split
is evaluated after selection, at the fixed 0.5 cutoff; accuracy, precision,
recall, F1, ROC-AUC, and a `[real, deepfake]` confusion matrix are saved to
`backend/artifacts/evaluation/celebdf_test_metrics.json`. Metrics must come from
a completed run; dataset-level results do not guarantee performance on newer
generators or every social-media compression path.

## Current validation status

`MODEL_EVALUATION_VALIDATED` remains false. The application displays the
checkpoint's thresholded prediction and its two raw class scores, but marks the
checkpoint unvalidated. Neither the pilot nor current Keras checkpoint is
accurate enough to claim reliable real/fake classification. Its sigmoid score
is not a calibrated probability of authenticity.

The Hugging Face image detector previously used by a fallback had incorrect
class-score handling and was removed from the application's active inference
path. It must not be used as a substitute for this video model or its evaluation.
