"""Evaluate the configured Keras detector on a reproducible balanced test sample.

The script uses the application's frame extraction, face crop, sequence
preparation, and inference functions. It never trains or changes model weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.ml.inference import predict_video  # noqa: E402
from app.services.video_service import extract_frames  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    predicted = (scores >= 0.5).astype(np.int64)
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    auc = float(np.mean(
        (positive[:, None] > negative[None, :])
        + 0.5 * (positive[:, None] == negative[None, :])
    ))
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "fake_precision": precision,
        "fake_recall": recall,
        "fake_f1": f1,
        "roc_auc": auc,
        "confusion_matrix_actual_real_fake_predicted_real_fake": [[tn, fp], [fn, tp]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/celebdf_manifest.csv")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/Celeb-DF-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "backend/artifacts/evaluation/active_keras_test_sample.json")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if settings.inference_backend.lower() != "keras":
        raise RuntimeError("Set INFERENCE_BACKEND=keras to evaluate the active Keras model")

    with args.manifest.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    by_label: dict[int, list[dict[str, str]]] = {0: [], 1: []}
    for row in rows:
        if row["split"] == "test":
            path = args.dataset_root / row["relative_path"]
            if path.is_file():
                by_label[int(row["label"])].append(row)

    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for label in (0, 1):
        rng.shuffle(by_label[label])
        if len(by_label[label]) < args.per_class:
            raise RuntimeError(f"Not enough held-out examples for label {label}")
        selected.extend(by_label[label][: args.per_class])

    cases = []
    for row in selected:
        path = args.dataset_root / row["relative_path"]
        frames = extract_frames(
            path,
            stride=settings.frame_stride,
            max_frames=settings.sequence_length * 4,
        )
        result = predict_video(frames)
        cases.append({
            "relative_path": row["relative_path"],
            "actual_label": "deepfake" if int(row["label"]) else "real",
            "fake_score": result["fakeProbability"],
            "predicted_label": "deepfake" if result["result"] == "FAKE" else "real",
        })

    labels = np.asarray([int(row["label"]) for row in selected], dtype=np.int64)
    scores = np.asarray([case["fake_score"] for case in cases], dtype=np.float64)
    report = {
        "checkpoint": str(settings.keras_model_artifact_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(settings.keras_model_artifact_path),
        "manifest": str(args.manifest.resolve().relative_to(ROOT)),
        "manifest_sha256": sha256(args.manifest),
        "dataset_split": "Celeb-DF v2 official test split",
        "sample_seed": args.seed,
        "samples_per_class": args.per_class,
        "label_mapping": {"0": "real", "1": "deepfake"},
        "score_semantics": "sigmoid output is interpreted as P(deepfake), matching training labels",
        "threshold": 0.5,
        "runtime_versions": {
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__,
            "opencv": cv2.__version__,
            "mtcnn": version("mtcnn"),
        },
        "metrics": metrics(labels, scores),
        "cases": cases,
        "warning": "Balanced sample only; not the full official test split and not evidence of reliable generalization.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved evaluation: {args.output}")


if __name__ == "__main__":
    main()
