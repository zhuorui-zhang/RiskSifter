import pytest
from risksifter.evaluation import evaluate_events


def test_duplicate_predictions_cannot_inflate_recall():
    refs = [{"video_id": "a", "duration": 20, "events": [{"start": 5, "end": 10}]}]
    predictions = [{"video_id": "a", "start": 5, "end": 10, "score": .8},
                   {"video_id": "a", "start": 5, "end": 10, "score": .7}]
    result = evaluate_events(predictions, refs)
    assert result["true_positives"] == 1 and result["false_positives"] == 1
    assert result["precision"] == .5 and result["recall"] == 1


def test_empty_negatives_and_misses():
    negative = [{"video_id": "n", "duration": 10, "events": []}]
    result = evaluate_events([], negative)
    assert result["recall"] is None and result["precision"] is None
    assert evaluate_events([{"video_id": "n", "start": 1, "end": 5}], negative)["false_positives"] == 1
    refs = [{"video_id": "p", "duration": 10, "events": [{"start": 2, "end": 4}]}]
    assert evaluate_events([], refs)["false_negatives"] == 1
    with pytest.raises(ValueError, match="missing"):
        evaluate_events([{"video_id": "unknown", "start": 1, "end": 5}], negative)
