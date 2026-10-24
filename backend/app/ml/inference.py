"""Inference for the project's configured trained CNN-LSTM artifact."""

import logging

import numpy as np
from fastapi import HTTPException, status

from app.core.config import settings
from app.ml.preprocessing import detect_and_crop_faces, make_sequence_tensor

logger = logging.getLogger(__name__)
_model_session = None
_model_input_name: str | None = None


class ModelShapeError(ValueError):
    """Raised when a loaded model does not match configured tensor shapes."""


def load_model():
    """Load and validate the explicitly configured trained model artifact."""
    global _model_session, _model_input_name
    if _model_session is not None:
        return _model_session
    backend = settings.inference_backend.lower()
    if backend not in {"onnx", "keras"}:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unsupported inference backend. Configure INFERENCE_BACKEND=keras or onnx for the trained CNN-LSTM model.",
        )
    model_path = (
        settings.keras_model_artifact_path
        if backend == "keras"
        else settings.onnx_model_artifact_path
    )
    if not model_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Trained CNN-LSTM model is unavailable. Check {backend.upper()} model path ({model_path}).",
        )
    try:
        expected_tail = (
            settings.sequence_length,
            settings.image_size,
            settings.image_size,
            3,
        )
        if backend == "keras":
            import tensorflow as tf

            model = tf.keras.models.load_model(str(model_path), compile=False)
            actual_shape = model.input_shape
            output_shape = model.output_shape
            if (
                not isinstance(actual_shape, tuple)
                or len(actual_shape) != 5
                or tuple(actual_shape[1:]) != expected_tail
            ):
                raise ModelShapeError(
                    f"Expected Keras input (batch,{expected_tail}), got {actual_shape}"
                )
            if (
                not isinstance(output_shape, tuple)
                or len(output_shape) != 2
                or output_shape[-1] != 1
            ):
                raise ModelShapeError(
                    f"Expected one sigmoid score per video, got {output_shape}"
                )
            _model_session = model
            _model_input_name = None
        else:
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            model_inputs = session.get_inputs()
            if len(model_inputs) != 1:
                raise ModelShapeError("CNN-LSTM ONNX model must have exactly one input")
            input_meta = model_inputs[0]
            actual_shape = input_meta.shape
            if len(actual_shape) != 5 or tuple(actual_shape[1:]) != expected_tail:
                raise ModelShapeError(
                    f"Expected ONNX input (batch,{expected_tail}), got {actual_shape}"
                )
            if len(session.get_outputs()) != 1:
                raise ModelShapeError("CNN-LSTM ONNX model must have exactly one output")
            _model_session = session
            _model_input_name = input_meta.name
    except HTTPException:
        raise
    except ModuleNotFoundError as exc:
        logger.exception("A required inference dependency is missing")
        if exc.name in {"tensorflow", "keras"}:
            detail = (
                "TensorFlow/Keras is not installed in the Python environment running the backend. "
                "Use Python 3.11, install backend/requirements.txt, and restart FastAPI."
            )
        else:
            detail = f"A required model-loading dependency is missing: {exc.name or 'unknown module'}."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        ) from exc
    except ModelShapeError as exc:
        logger.exception("Configured CNN-LSTM checkpoint has an incompatible model shape")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The configured CNN-LSTM checkpoint input/output shape does not match settings "
                f"(sequence={settings.sequence_length}, image={settings.image_size})."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Could not load configured CNN-LSTM model")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The configured CNN-LSTM checkpoint could not be deserialized. "
                "Check that the checkpoint matches the TensorFlow 2.20/Keras runtime "
                "and that the configured artifact path is correct."
            ),
        ) from exc
    return _model_session


def predict_video(frames: list[np.ndarray], filename: str | None = None) -> dict:
    if not frames:
        raise HTTPException(
            status_code=422,
            detail="No decodable video frames were found. Check that the file is a valid supported video.",
        )

    model = load_model()
    # Training used frames sampled at FRAME_STRIDE, MTCNN face crops, 224x224
    # RGB images, and [0,1] values. Preserve that exact preprocessing path.
    faces: list[np.ndarray] = []
    # Training keeps the first 20 detected face crops. Process candidate
    # frames in order and stop as soon as the trained sequence is complete.
    for start in range(0, len(frames), settings.sequence_length):
        batch_faces, _, _ = detect_and_crop_faces(
            frames[start : start + settings.sequence_length],
            target_size=settings.image_size,
        )
        faces.extend(batch_faces)
        if len(faces) >= settings.sequence_length:
            break
    if not faces:
        raise HTTPException(
            status_code=422,
            detail="No face was detected in the sampled frames, so the face-based model cannot analyze this video.",
        )

    sequence = make_sequence_tensor(faces)
    if sequence.shape != (
        1,
        settings.sequence_length,
        settings.image_size,
        settings.image_size,
        3,
    ):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Preprocessed frame sequence does not match the trained model input.",
        )

    try:
        raw_score = (
            model.run(None, {_model_input_name: sequence})[0]
            if hasattr(model, "run")
            else model(sequence, training=False)
        )
        scores = np.asarray(raw_score).reshape(-1)
        if scores.size != 1:
            raise ValueError(f"Expected one score for one video, received {scores.shape}")
        raw_val = float(scores[0])
    except Exception as exc:
        logger.exception("CNN-LSTM inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The CNN-LSTM model failed to analyze this video.",
        ) from exc

    if not np.isfinite(raw_val) or not 0.0 <= raw_val <= 1.0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The model returned an invalid sigmoid score; no classification was produced.",
        )

    # Inverted probability mapping to match training convention
    real_probability = raw_val
    fake_probability = 1.0 - real_probability
    inference_status = "validated" if settings.model_evaluation_validated else "unvalidated"
    result = "FAKE" if fake_probability >= 0.5 else "REAL"
    return {
        "modelMode": f"trained_cnn_lstm_{settings.inference_backend.lower()}",
        "inferenceStatus": inference_status,
        "result": result,
        "confidence": round(max(fake_probability, real_probability) * 100.0, 2),
        # Keep enough precision for ranking metrics and threshold reproducibility.
        "fakeProbability": round(fake_probability, 6),
        "realProbability": round(real_probability, 6),
        # This checkpoint emits a video-level sigmoid only, not frame logits.
        "suspiciousIndices": [],
        "timeline": [],
        "croppedFaces": [],
        "modelsComparison": None,
        "audioAnalysis": None,
        "facesAnalysis": None,
    }
