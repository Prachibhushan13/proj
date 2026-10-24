import numpy as np

from app.ml import inference


class FixedScoreSession:
    def __init__(self, score: float):
        self.score = score

    def run(self, _outputs, _inputs):
        return [np.array([[self.score]], dtype=np.float32)]


def run_prediction(monkeypatch, *, validated: bool, score: float):
    monkeypatch.setattr(inference.settings, "model_evaluation_validated", validated)
    monkeypatch.setattr(inference, "_model_session", FixedScoreSession(score))
    monkeypatch.setattr(inference, "_model_input_name", "input")
    monkeypatch.setattr(
        inference,
        "detect_and_crop_faces",
        lambda frames, target_size: (
            [np.zeros((target_size, target_size, 3), dtype=np.float32)], [], []
        ),
    )
    return inference.predict_video([np.zeros((8, 8, 3), dtype=np.uint8)])


def test_unvalidated_checkpoint_returns_its_thresholded_label_and_raw_score(monkeypatch):
    prediction = run_prediction(monkeypatch, validated=False, score=0.99)

    assert prediction["result"] == "FAKE"
    assert prediction["inferenceStatus"] == "unvalidated"
    assert prediction["fakeProbability"] == 0.99


def test_validated_checkpoint_uses_trained_label_mapping_and_cutoff(monkeypatch):
    prediction = run_prediction(monkeypatch, validated=True, score=0.99)

    assert prediction["result"] == "FAKE"
    assert prediction["inferenceStatus"] == "validated"


def test_low_score_maps_to_real_without_claiming_validation(monkeypatch):
    prediction = run_prediction(monkeypatch, validated=False, score=0.27)

    assert prediction["result"] == "REAL"
    assert prediction["inferenceStatus"] == "unvalidated"
    assert prediction["realProbability"] == 0.73
