from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "Celeb-DF-v2"
TEST_LIST = DATASET_ROOT / "List_of_testing_videos.txt"
OUTPUT = ROOT / "datasets" / "celebdf_manifest.csv"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

FOLDER_LABELS = {
    "Celeb-real": 0,
    "YouTube-real": 0,
    "Celeb-synthesis": 1,
}


def main():
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET_ROOT}")
    if not TEST_LIST.is_file():
        raise FileNotFoundError(f"Official test list not found: {TEST_LIST}")

    official_test = set()

    for line in TEST_LIST.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue

        list_label, relative_path = parts
        relative_path = relative_path.strip()
        folder = Path(relative_path).parts[0]

        # Celeb-DF's list uses 1=real and 0=fake.
        expected_list_label = 1 if FOLDER_LABELS.get(folder) == 0 else 0
        if list_label != str(expected_list_label):
            raise ValueError(
                f"Label mismatch in official test list: {line}"
            )

        official_test.add(relative_path)

    rows = []
    for folder, label in FOLDER_LABELS.items():
        for path in sorted((DATASET_ROOT / folder).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            relative_path = path.relative_to(DATASET_ROOT).as_posix()
            rows.append({
                "video_path": str(path.resolve()),
                "relative_path": relative_path,
                "folder": folder,
                "label": label,
                "split": "test" if relative_path in official_test else "pending",
            })

    manifest = pd.DataFrame(rows)

    if manifest.empty:
        raise RuntimeError("No videos found.")
    if manifest["relative_path"].duplicated().any():
        raise RuntimeError("Duplicate video paths found.")

    missing_test = official_test - set(manifest["relative_path"])
    if missing_test:
        raise RuntimeError(
            f"{len(missing_test)} official test videos are missing: "
            f"{sorted(missing_test)[:10]}"
        )

    remaining = manifest[manifest["split"] == "pending"].copy()
    train_df, val_df = train_test_split(
        remaining,
        test_size=0.15,
        stratify=remaining["label"],
        random_state=42,
    )

    manifest.loc[train_df.index, "split"] = "train"
    manifest.loc[val_df.index, "split"] = "val"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = manifest.sort_values(["split", "relative_path"])
    manifest.to_csv(OUTPUT, index=False)

    print(f"Manifest saved: {OUTPUT}")
    print(f"Total videos: {len(manifest)}")
    print(f"Official test videos matched: {len(official_test)}")
    print("\nCounts by split and label (0=real, 1=fake):")
    print(pd.crosstab(manifest["split"], manifest["label"], margins=True))
    print("\nCounts by folder and split:")
    print(pd.crosstab(manifest["folder"], manifest["split"], margins=True))


if __name__ == "__main__":
    main()
