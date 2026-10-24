from collections.abc import Iterable

import cv2
import numpy as np

from app.core.config import settings

try:
    from mtcnn import MTCNN
    detector = MTCNN()
except Exception:
    detector = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def align_face(image: np.ndarray, keypoints: dict) -> np.ndarray:
    left_eye = keypoints.get("left_eye")
    right_eye = keypoints.get("right_eye")
    if not left_eye or not right_eye:
        return image
    
    # Calculate angle between eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Rotate around the center of the eyes
    eye_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    return rotated


def detect_and_crop_faces(frames: Iterable[np.ndarray], target_size: int = settings.image_size, is_xception: bool = False, return_pil: bool = False) -> tuple[list[np.ndarray], list[int], list[dict]]:
    faces = []
    suspicious_indices: list[int] = []
    all_keypoints: list[dict] = []
    for index, frame in enumerate(frames):
        detections = []
        if detector is not None:
            try:
                detections = detector.detect_faces(frame)
            except Exception:
                detections = []

        if not detections and not face_cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            for (x, y, w, h) in rects:
                detections.append({
                    "box": [int(x), int(y), int(w), int(h)],
                    "confidence": 0.96,
                    "keypoints": {
                        "left_eye": [int(x + w * 0.35), int(y + h * 0.35)],
                        "right_eye": [int(x + w * 0.65), int(y + h * 0.35)],
                    },
                })

        if not detections:
            continue
        
        # Align face before cropping to stabilize forensic signals
        aligned_frame = align_face(frame, detections[0].get("keypoints", {}))
        
        # Re-detect on aligned frame for precise cropping
        re_detections = []
        if detector is not None:
            try:
                re_detections = detector.detect_faces(aligned_frame)
            except Exception:
                re_detections = []
        
        target_det = re_detections[0] if re_detections else detections[0]
        x, y, width, height = target_det["box"]
        # Expand crop box slightly (margin)
        margin = int(width * 0.1)
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(width + margin * 2, aligned_frame.shape[1] - x)
        h = min(height + margin * 2, aligned_frame.shape[0] - y)
        
        crop = aligned_frame[y : y + h, x : x + w]
        if crop.size == 0:
            continue
            
        if return_pil:
            from PIL import Image
            faces.append(Image.fromarray(crop))
        else:
            resized = cv2.resize(crop, (target_size, target_size))
            if is_xception:
                # Xception models strictly require [-1, 1] scaling. 
                # [0, 1] scaling causes the model to guess randomly (~0.5).
                faces.append((resized.astype("float32") / 127.5) - 1.0)
            else:
                faces.append(resized.astype("float32") / 255.0)
            
        all_keypoints.append(target_det.get("keypoints", {}))
        if target_det.get("confidence", 0) < 0.95:
            suspicious_indices.append(index)
    return faces, suspicious_indices, all_keypoints


def make_sequence_tensor(faces: list[np.ndarray]) -> np.ndarray:
    if not faces:
        return np.zeros((1, settings.sequence_length, settings.image_size, settings.image_size, 3), dtype="float32")
    sequence = faces[: settings.sequence_length]
    while len(sequence) < settings.sequence_length:
        sequence.append(sequence[-1])
    return np.expand_dims(np.stack(sequence, axis=0), axis=0)

