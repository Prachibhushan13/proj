from bson import ObjectId
from bson.errors import InvalidId

from app.core.config import settings
from app.db.mongodb import predictions_collection
from app.ml.inference import predict_video
from app.models.prediction import PredictionInDB
from app.services.video_service import extract_frames, save_upload
from app.utils.serialization import serialize_document

UNTRUSTED_LEGACY_MODES = {
    "mock",
    "forensic_backbone",
    "legacy",
    "trained",
    "transformers_vit",
    "trained_xception",
}


def normalize_prediction(document: dict) -> dict:
    normalized = serialize_document(document)
    normalized.setdefault("modelMode", "legacy")
    normalized.setdefault("inferenceStatus", "unvalidated")
    normalized.setdefault("suspiciousFrames", [])
    normalized.setdefault("heatmapFrames", [])
    normalized.setdefault("frameTimeline", [])
    if normalized.get("modelMode") in UNTRUSTED_LEGACY_MODES:
        # Do not continue displaying old heuristic, random, inverted-label,
        # or third-party fallback scores as valid classifications.
        normalized.update(
            {
                "modelMode": "legacy_unverified",
                "inferenceStatus": "unvalidated",
                "result": "INCONCLUSIVE",
                "confidence": 0.0,
                "realProbability": 0.5,
                "fakeProbability": 0.5,
                "suspiciousFrames": [],
                "heatmapFrames": [],
                "frameTimeline": [],
                "modelsComparison": None,
                "facesAnalysis": None,
                "audioAnalysis": None,
            }
        )
    return normalized


def analyze_video(file, user_id: str) -> dict:
    saved_path = save_upload(file)
    # Avoid lossy transcoding before forensic inference; training reads the
    # original decoded frames and compression can erase or add artifacts.
    # The model uses the first 20 detected face crops; provide a bounded
    # candidate window so long uploads do not consume unbounded memory.
    frames = extract_frames(
        saved_path,
        max_frames=settings.sequence_length * 4,
    )
    inference = predict_video(frames, filename=file.filename)
    # The active temporal model emits a single video score, not frame-level
    # evidence. Do not manufacture highlighted frames or heatmap assets.
    suspicious_paths: list[str] = []
    heatmap_paths: list[str] = []
    document = PredictionInDB(
        userId=user_id,
        filename=file.filename,
        modelMode=inference["modelMode"],
        inferenceStatus=inference["inferenceStatus"],
        result=inference["result"],
        confidence=inference["confidence"],
        realProbability=inference["realProbability"],
        fakeProbability=inference["fakeProbability"],
        suspiciousFrames=suspicious_paths,
        heatmapFrames=heatmap_paths,
        frameTimeline=inference["timeline"],
        modelsComparison=inference.get("modelsComparison"),
        facesAnalysis=inference.get("facesAnalysis"),
        audioAnalysis=inference.get("audioAnalysis"),
    )
    inserted = predictions_collection.insert_one(document.model_dump())
    created = predictions_collection.find_one({"_id": inserted.inserted_id})
    return normalize_prediction(created)


def get_prediction_history(user_id: str) -> list[dict]:
    cursor = predictions_collection.find({"userId": user_id}).sort("createdAt", -1)
    return [normalize_prediction(item) for item in cursor]


def get_prediction_by_id(prediction_id: str, user_id: str | None = None) -> dict | None:
    try:
        query = {"_id": ObjectId(prediction_id)}
    except InvalidId:
        return None
    if user_id:
        query["userId"] = user_id
    item = predictions_collection.find_one(query)
    return normalize_prediction(item) if item else None


def delete_prediction(prediction_id: str, user_id: str) -> bool:
    try:
        object_id = ObjectId(prediction_id)
    except InvalidId:
        return False
    deleted = predictions_collection.delete_one({"_id": object_id, "userId": user_id})
    return deleted.deleted_count > 0
