from visionguard.analytics.engine import AnalyticsEngine
from visionguard.domain import Detection, EventType, Track, ZoneRule


def track(track_id, x, y):
    detection = Detection((x - 5, y - 5, x + 5, y + 5), .9, 0, "person")
    return Track(track_id, detection, [(x, y)])


def test_zone_enter_and_exit():
    zone = ZoneRule("restricted", [(100, 100), (200, 100), (200, 200), (100, 200)], {"person"})
    engine = AnalyticsEngine("cam", [zone])
    assert engine.process([track(1, 50, 50)]) == []
    entered = engine.process([track(1, 150, 150)])
    exited = engine.process([track(1, 250, 250)])
    assert entered[0].event_type == EventType.ZONE_ENTER
    assert exited[0].event_type == EventType.ZONE_EXIT
