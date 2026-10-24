# Training Guide

Dataset source used for the pilot: Kaggle `reubensuju/celeb-df-v2` (Celeb-DF
v2). The Kaggle listing does not specify a reuse license; keep the downloaded
archive local and check the dataset terms before redistributing it.

1. Extract Celeb-DF v2 to `datasets/Celeb-DF-v2/` (the official directory
   structure and `List_of_testing_videos.txt` must be preserved).
2. Generate a video-level manifest. The official test list remains held out;
   the remaining videos are split into training and validation sets:

```bash
python scripts/prepare_celebdf.py
```

3. Train the model:

```bash
python scripts/train_model.py --manifest datasets/celebdf_manifest.csv \
  --dataset-root datasets/Celeb-DF-v2 --epochs 20 --batch-size 2
```

For a bounded smoke/evaluation run on a CPU, add
`--max-per-class-per-split 40 --freeze-backbone`. This samples at most 40
videos of each class separately from train, validation, and the official
held-out test split, and trains the temporal/classification layers while
keeping ImageNet EfficientNet features fixed. The input sequences are cached in
memory after decoding to avoid repeating face detection every epoch. The
resulting metrics are for that smaller sample only and must not be presented
as full-dataset performance. Remove the cap for the full manifest; omit
`--freeze-backbone` to fine-tune the final 20 EfficientNet layers. Add
`--cache-sequences` only for a capped run; caching the full dataset in RAM can
require tens of gigabytes.

The training pipeline includes:

- EfficientNetB0 transfer learning backbone
- Bidirectional LSTM temporal modeling
- Early stopping
- ReduceLROnPlateau
- Model checkpoints
- TensorBoard logging
- Mixed precision policy

The generator uses the same frame stride, face detector/cropper, RGB resize,
and 20-face padding as application inference. Keras EfficientNet expects
0–255 pixels; the model now rescales the generator's 0–1 face tensors inside
the graph. The training loss compensates for class imbalance. Metrics are
computed once on the manifest's held-out `test` split at the fixed 0.5 cutoff
and saved to `backend/artifacts/evaluation/celebdf_test_metrics.json`.

Optional export tooling:

```bash
pip install -r backend/requirements-export.txt
```
