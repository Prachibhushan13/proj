"""Train and evaluate the project's face-sequence CNN-LSTM model."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.core.config import settings
from app.ml.model import build_cnn_lstm_model
from app.ml.preprocessing import detect_and_crop_faces, make_sequence_tensor
from app.services.video_service import extract_frames


class VideoSequenceGenerator(tf.keras.utils.Sequence):
    """Decode videos lazily using the same frame/face preprocessing as inference."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        batch_size: int = 2,
        *,
        shuffle: bool = False,
        dataset_root: str | Path = "datasets/Celeb-DF-v2",
        cache_sequences: bool = False,
    ) -> None:
        super().__init__()
        self.manifest = manifest.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_root = Path(dataset_root)
        self.cache_sequences = cache_sequences
        self._sequence_cache: dict[int, np.ndarray] = {}
        self.order = np.arange(len(self.manifest))
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.manifest) / self.batch_size)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.order)

    def _video_path(self, row) -> Path:
        source = Path(str(getattr(row, "video_path", "")))
        if source.is_file():
            return source
        relative = Path(str(row.relative_path))
        candidate = self.dataset_root / relative
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"Video not found at manifest path {source} or dataset path {candidate}"
        )

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        positions = self.order[index * self.batch_size : (index + 1) * self.batch_size]
        batch = self.manifest.iloc[positions]
        sequences: list[np.ndarray] = []
        labels: list[float] = []
        for local_index, row in enumerate(batch.itertuples(index=False)):
            cache_key = int(positions[local_index])
            if self.cache_sequences and cache_key in self._sequence_cache:
                sequences.append(self._sequence_cache[cache_key])
                labels.append(float(row.label))
                continue
            video_path = self._video_path(row)
            frames = extract_frames(
                video_path,
                stride=settings.frame_stride,
                max_frames=settings.sequence_length * 4,
            )
            if not frames:
                raise ValueError(f"No decodable frames in {video_path}")
            faces, _, _ = detect_and_crop_faces(
                frames,
                target_size=settings.image_size,
            )
            if not faces:
                raise ValueError(f"No face detected in training video {video_path}")
            sequence = make_sequence_tensor(faces)[0]
            sequences.append(sequence)
            if self.cache_sequences:
                self._sequence_cache[cache_key] = sequence
            labels.append(float(row.label))
        return (
            np.asarray(sequences, dtype=np.float32),
            np.asarray(labels, dtype=np.float32),
        )


def train_model(
    manifest_path: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 2,
    dataset_root: str | Path = "datasets/Celeb-DF-v2",
    max_per_class_per_split: int | None = None,
    freeze_backbone: bool = False,
    cache_sequences: bool = False,
    evaluation_output: str | Path | None = None,
) -> dict:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if max_per_class_per_split is not None and max_per_class_per_split < 1:
        raise ValueError("max_per_class_per_split must be positive when supplied")
    manifest = pd.read_csv(manifest_path)
    required = {"relative_path", "label", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    labels = set(manifest["label"].dropna().astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValueError("Labels must use 0=real and 1=fake")

    manifest_video_count = len(manifest)
    dataset_root = Path(dataset_root)
    available_video_count = manifest["relative_path"].map(
        lambda path: (dataset_root / str(path)).is_file()
    )
    if max_per_class_per_split is not None:
        # Partial downloads are useful for a bounded run only when their actual
        # files are filtered before sampling. The official split assignments
        # remain unchanged; absent videos are never substituted or relabeled.
        manifest = manifest[available_video_count].copy()

    train_df = manifest[manifest["split"] == "train"].copy()
    val_df = manifest[manifest["split"] == "val"].copy()
    test_df = manifest[manifest["split"] == "test"].copy()
    if max_per_class_per_split is not None:
        # A reproducible quick run for constrained machines. Keep the official
        # test split isolated, and sample each class independently so a small
        # run does not inherit the dataset's substantial class imbalance.
        capped_splits = []
        for split_df in (train_df, val_df, test_df):
            sampled_classes = [
                group.sample(
                    n=min(len(group), max_per_class_per_split),
                    random_state=42,
                )
                for _, group in split_df.groupby("label")
            ]
            capped_splits.append(
                pd.concat(sampled_classes).sort_index()
                if sampled_classes
                else split_df
            )
        train_df, val_df, test_df = capped_splits
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Manifest must contain non-empty train, val, and test splits")
    tf.keras.utils.set_random_seed(42)

    # Mixed precision is useful on a supported GPU but tends to reduce
    # throughput and stability on this project's CPU-only local environment.
    has_gpu = bool(tf.config.list_physical_devices("GPU"))
    tf.keras.mixed_precision.set_global_policy("mixed_float16" if has_gpu else "float32")
    settings.model_artifact_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_cnn_lstm_model(train_backbone=not freeze_backbone)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=6,
            monitor="val_loss",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(settings.model_artifact_path),
            save_best_only=True,
            monitor="val_loss",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=3,
            factor=0.2,
            monitor="val_loss",
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(settings.artifact_path / "logs")
        ),
    ]
    train_gen = VideoSequenceGenerator(
        train_df,
        batch_size,
        shuffle=True,
        dataset_root=dataset_root,
        cache_sequences=cache_sequences,
    )
    val_gen = VideoSequenceGenerator(
        val_df,
        batch_size,
        dataset_root=dataset_root,
        cache_sequences=cache_sequences,
    )
    class_counts = train_df["label"].astype(int).value_counts().to_dict()
    if set(class_counts) != {0, 1}:
        raise ValueError("Training split must contain both real and fake videos")
    total_train = len(train_df)
    class_weight = {
        int(label): total_train / (2.0 * count)
        for label, count in class_counts.items()
    }
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # The held-out test split is evaluated once, after checkpoint selection;
    # it is not used to tune the cutoff or select model weights.
    test_gen = VideoSequenceGenerator(
        test_df,
        batch_size,
        dataset_root=dataset_root,
        cache_sequences=cache_sequences,
    )
    probabilities = model.predict(test_gen, verbose=0).reshape(-1)
    y_true = test_df["label"].to_numpy(dtype=np.int32)
    if len(probabilities) != len(y_true):
        raise RuntimeError(
            f"Test generator returned {len(probabilities)} predictions for {len(y_true)} labels"
        )
    predictions = (probabilities >= 0.5).astype(np.int32)
    metrics = {
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": hashlib.sha256(
            Path(manifest_path).read_bytes()
        ).hexdigest(),
        "manifest_videos_total": int(manifest_video_count),
        "manifest_videos_available": int(available_video_count.sum()),
        "training_videos": int(len(train_df)),
        "validation_videos": int(len(val_df)),
        "test_videos": int(len(y_true)),
        "real_videos": int((y_true == 0).sum()),
        "fake_videos": int((y_true == 1).sum()),
        "threshold": 0.5,
        "evaluation_scope": (
            "balanced_pilot_sample"
            if max_per_class_per_split is not None
            else "manifest_test_split"
        ),
        "evaluation_warning": (
            "Small capped sample; do not interpret as full-dataset performance."
            if max_per_class_per_split is not None
            else None
        ),
        "label_mapping": {"0": "real", "1": "deepfake"},
        "class_weights": {str(key): float(value) for key, value in class_weight.items()},
        "epochs_requested": int(epochs),
        "epochs_completed": int(len(history.epoch)),
        "batch_size": int(batch_size),
        "max_per_class_per_split": max_per_class_per_split,
        "freeze_backbone": bool(freeze_backbone),
        "cache_sequences": bool(cache_sequences),
        "checkpoint": str(settings.model_artifact_path.resolve()),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["real", "deepfake"],
            output_dict=True,
            zero_division=0,
        ),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, probabilities))
        if len(np.unique(y_true)) == 2
        else None
    )
    evaluation_path = (
        Path(evaluation_output)
        if evaluation_output is not None
        else settings.artifact_path / "evaluation" / "celebdf_test_metrics.json"
    )
    if not evaluation_path.is_absolute():
        evaluation_path = settings.project_root / evaluation_path
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["evaluation_file"] = str(evaluation_path)
    return metrics
