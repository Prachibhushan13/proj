from app.services.prediction_service import normalize_prediction


def test_old_transformer_result_is_not_presented_as_a_valid_verdict() -> None:
    normalized = normalize_prediction(
        {
            "_id": "legacy-row-1",
            "modelMode": "transformers_vit",
            "result": "FAKE",
            "confidence": 99.0,
            "fakeProbability": 0.99,
            "realProbability": 0.01,
            "frameTimeline": [{"frameIndex": 0, "score": 0.84}],
            "suspiciousFrames": ["/static/old-frame.jpg"],
        }
    )

    assert normalized["modelMode"] == "legacy_unverified"
    assert normalized["result"] == "INCONCLUSIVE"
    assert normalized["confidence"] == 0.0
    assert normalized["fakeProbability"] == 0.5
    assert normalized["realProbability"] == 0.5
    assert normalized["frameTimeline"] == []
    assert normalized["suspiciousFrames"] == []
