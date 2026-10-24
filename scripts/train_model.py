import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "backend"))

from app.ml.train import train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the CNN-LSTM from a labeled video manifest."
    )
    parser.add_argument("--manifest", default="datasets/celebdf_manifest.csv")
    parser.add_argument("--dataset-root", default="datasets/Celeb-DF-v2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--max-per-class-per-split",
        type=int,
        default=None,
        help=(
            "Optional reproducible cap for quick runs; sample up to this many "
            "videos per class independently in train, validation, and official test."
        ),
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train the temporal/classification layers while freezing EfficientNetB0.",
    )
    parser.add_argument(
        "--cache-sequences",
        action="store_true",
        help="Keep decoded sequences in RAM between epochs (use only for capped runs).",
    )
    parser.add_argument(
        "--evaluation-output",
        default=None,
        help="Optional JSON metrics path; defaults to the standard Celeb-DF report path.",
    )
    args = parser.parse_args()
    metrics = train_model(
        args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_root=args.dataset_root,
        max_per_class_per_split=args.max_per_class_per_split,
        freeze_backbone=args.freeze_backbone,
        cache_sequences=args.cache_sequences,
        evaluation_output=args.evaluation_output,
    )
    print(json.dumps(metrics, indent=2))
