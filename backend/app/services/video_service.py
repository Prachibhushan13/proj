import shutil
import subprocess
import uuid
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.core.config import settings

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
logger = logging.getLogger(__name__)


def ensure_directories() -> None:
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    (settings.upload_path / "frames").mkdir(parents=True, exist_ok=True)
    (settings.upload_path / "faces").mkdir(parents=True, exist_ok=True)
    (settings.upload_path / "reports").mkdir(parents=True, exist_ok=True)
    (settings.artifact_path / "models").mkdir(parents=True, exist_ok=True)


def validate_upload(file: UploadFile) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported video format")


def save_upload(file: UploadFile) -> Path:
    ensure_directories()
    validate_upload(file)
    file.file.seek(0, 2)
    size_bytes = file.file.tell()
    file.file.seek(0)
    if size_bytes > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video exceeds the {settings.max_upload_mb} MB upload limit",
        )
    safe_name = Path(file.filename or "video").name
    target = settings.upload_path / f"{uuid.uuid4()}_{safe_name}"
    with target.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return target


def compress_video(input_path: Path) -> Path:
    output_path = input_path.with_name(f"compressed_{input_path.name}")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vcodec",
        "libx264",
        "-crf",
        "28",
        str(output_path),
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True)
    except FileNotFoundError:
        logger.warning("ffmpeg not found. Skipping compression and using original upload.")
        return input_path
    if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    logger.warning("ffmpeg could not transcode %s; attempting original upload", input_path.name)
    return input_path


def extract_frames(
    video_path: Path,
    stride: int | None = None,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    sample_full_video = stride is None and (
        settings.inference_backend.lower() == "torch"
        and settings.torch_temporal_sampling == "full_video"
    )
    if stride is None:
        stride = settings.frame_stride
    if stride < 1:
        raise ValueError("Frame stride must be positive")
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    if not capture.isOpened():
        capture.release()
        return frames
    if sample_full_video:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            # Read only the sequence required by the model instead of keeping
            # every decoded frame from a long upload in memory.
            for frame_index in np.linspace(
                0, frame_count - 1, num=settings.sequence_length, dtype=np.int64
            ):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                success, frame = capture.read()
                if not success:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            capture.release()
            return frames
    frame_idx = 0
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        if frame_idx % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames is not None and len(frames) >= max_frames:
                break
        frame_idx += 1
    capture.release()
    return frames
