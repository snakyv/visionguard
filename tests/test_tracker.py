from visionguard.domain import Detection
from visionguard.tracking.centroid import CentroidTracker


def test_centroid_tracker_keeps_identity():
    tracker = CentroidTracker(max_distance=50)
    first = tracker.update([Detection((10, 10, 30, 50), .9, 0, "person")])
    second = tracker.update([Detection((18, 12, 38, 52), .9, 0, "person")])
    assert first[0].track_id == second[0].track_id
    assert len(second[0].history) == 2
