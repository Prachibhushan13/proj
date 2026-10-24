# PyTorch Celeb-DF Temporal Training

This pipeline trains only the temporal projection, bidirectional LSTMs, and
classifier on cached EfficientNetB0 features. It does not recompute the
backbone during training and does not modify the existing TensorFlow/Keras
artifacts.

## Environment and data

Use the project virtual environment (or the documented WSL environment):

```bash
source ~/.venvs/deepfake-torch/bin/activate
```

The manifest uses `0 = real` and `1 = fake`. The official `test` split is
never used for optimization, early stopping, or threshold selection.

## Validate the cache

```bash
python scripts/validate_celebdf_cache.py
```

The command checks index/manifest uniqueness, labels and splits, file
existence, `(20, 1280)` shape, finite values, and orphan files. It returns a
nonzero status when any issue is found. Orphan files are reported but not
deleted.

## Train and resume

```bash
python scripts/train_torch_temporal.py --epochs 20 --batch-size 32
python scripts/train_torch_temporal.py \
  --resume backend/artifacts/models/torch/latest_checkpoint.pt
```

Training uses `BCEWithLogitsLoss` with a positive-class weight derived from
the training split. It reports loss, accuracy, precision, recall, F1, ROC-AUC,
and a confusion matrix for validation. Artifacts are written to
`backend/artifacts/models/torch/`:

- `best_checkpoint.pt`
- `latest_checkpoint.pt`
- `training_history.csv`
- `validation_metrics.json`
- `training_config.json`

## Evaluate once on the official test split

```bash
python scripts/evaluate_torch_temporal.py \
  --checkpoint backend/artifacts/models/torch/best_checkpoint.pt
```

This uses the fixed 0.5 decision threshold and writes `test_metrics.json` and
`test_metrics.npz` beside the selected checkpoint. The NPZ contains raw
probabilities, labels, and relative paths. Do not tune the threshold on this
split.

## Application integration

This describes the experimental PyTorch checkpoint pipeline only. The active
application uses the Keras CNN-LSTM checkpoint documented in
[`MODEL_NOTES.md`](MODEL_NOTES.md). The PyTorch checkpoint is not selected by
the current API configuration.

To run the experimental adapter in a separate setup, configure:

```env
INFERENCE_BACKEND=torch
TORCH_MODEL_PATH=backend/artifacts/models/torch/best_checkpoint.pt
```

The adapter loads the checkpoint once on first inference, selects CUDA when
available, and uses the same 20-frame, RGB, 224x224, `[0, 1]` input convention
and ImageNet normalization as the PyTorch model. If the configured checkpoint
is missing or cannot be loaded, the API returns a model-unavailable error and
does not substitute a different or untrained model. `fakeProbability` is a
model score and is not a calibrated probability. The current model emits only
a video-level score, not per-frame predictions.

## Limitations and troubleshooting

- The current split is video-level and has not been proven identity-disjoint.
- Training and evaluation require a compatible PyTorch/torchvision install;
  the repository's TensorFlow dependencies are intentionally unchanged.
- An 8 GB GPU should use modest batches. Reduce `--batch-size` if memory is
  exhausted.
- A CPU-only environment can run the smoke tests but is not evidence of a
  completed full training run.
- Missing or invalid cache entries must be repaired before training.
