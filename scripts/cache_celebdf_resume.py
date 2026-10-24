from pathlib import Path
import hashlib

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets" / "celebdf_manifest.csv"
CACHE = ROOT / "datasets" / "celebdf_features"
INDEX = CACHE / "features_index.csv"

T, SIZE, FEATURE_SIZE, BATCH = 20, 224, 1280, 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cache_name(relative_path):
    key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
    return f"{key}.npy"


def extract_frames_sequentially(path):
    """Decode sequentially and select 20 evenly spaced frames."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")

        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1:
            raise RuntimeError("Video has no reported frames")

        targets = np.linspace(0, count - 1, T, dtype=np.int64)
        targets_by_frame = {}
        for output_index, frame_index in enumerate(targets):
            targets_by_frame.setdefault(int(frame_index), []).append(output_index)

        frames = [None] * T
        frame_index = 0

        while frame_index <= int(targets[-1]):
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index in targets_by_frame:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (SIZE, SIZE))
                frame = frame.astype(np.float32) / 255.0

                for output_index in targets_by_frame[frame_index]:
                    frames[output_index] = frame

            frame_index += 1

        if any(frame is None for frame in frames):
            raise RuntimeError(
                f"Could not decode all selected frames "
                f"({sum(frame is not None for frame in frames)}/{T})"
            )

        return np.stack(frames)
    finally:
        cap.release()


def valid_cache(path):
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == (T, FEATURE_SIZE) and np.isfinite(arr).all()
    except Exception:
        return False


def main():
    df = pd.read_csv(MANIFEST)
    required = {"video_path", "relative_path", "label", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain: {sorted(required)}")

    CACHE.mkdir(parents=True, exist_ok=True)
    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    backbone = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )
    backbone.classifier = nn.Identity()
    backbone = backbone.to(DEVICE).eval()

    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=DEVICE
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=DEVICE
    ).view(1, 3, 1, 1)

    records, failures = [], []
    reused = newly_cached = 0

    def save_index():
        pd.DataFrame(records).to_csv(INDEX, index=False)

    try:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            rel = str(row.relative_path)
            output = CACHE / cache_name(rel)
            status = "ok"
            error = ""

            try:
                if valid_cache(output):
                    reused += 1
                else:
                    frames = extract_frames_sequentially(row.video_path)
                    tensor = torch.from_numpy(frames).permute(
                        0, 3, 1, 2
                    ).to(DEVICE)
                    tensor = (tensor - mean) / std

                    chunks = []
                    with torch.inference_mode():
                        for start in range(0, T, BATCH):
                            chunks.append(
                                backbone(tensor[start:start + BATCH]).cpu()
                            )

                    features = torch.cat(chunks).numpy().astype(np.float32)
                    if features.shape != (T, FEATURE_SIZE):
                        raise RuntimeError(f"Unexpected shape {features.shape}")
                    if not np.isfinite(features).all():
                        raise RuntimeError("Non-finite feature values")

                    temp = output.with_suffix(".tmp.npy")
                    np.save(temp, features)
                    temp.replace(output)
                    newly_cached += 1

            except Exception as exc:
                status = "failed"
                error = str(exc)
                failures.append((rel, error))
                print(f"\nFAILED [{i}/{len(df)}] {rel}: {error}")

            records.append({
                "relative_path": rel,
                "feature_path": str(output) if status == "ok" else "",
                "label": int(row.label),
                "split": row.split,
                "status": status,
                "error": error,
            })

            if i % 25 == 0 or i == len(df):
                save_index()
                print(
                    f"Processed {i}/{len(df)} | "
                    f"new: {newly_cached} | reused: {reused} | "
                    f"failed: {len(failures)}",
                    flush=True,
                )

    except KeyboardInterrupt:
        save_index()
        print("\nInterrupted safely. Progress index saved.")
        print("Run this same script again to resume.")
        return

    save_index()
    print("\n--- Summary ---")
    print("Manifest rows:", len(df))
    print("Newly cached:", newly_cached)
    print("Reused valid caches:", reused)
    print("Failed:", len(failures))
    print("Index:", INDEX)

    if failures:
        print("Failed video paths are recorded in the index CSV.")


if __name__ == "__main__":
    main()
