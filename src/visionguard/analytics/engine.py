from __future__ import annotations

from collections import defaultdict
from math import hypot
from time import monotonic

from visionguard.analytics.geometry import point_in_polygon, side_of_line
from visionguard.domain import AnalyticsEvent, EventType, LineRule, Track, ZoneRule


class AnalyticsEngine:
    def __init__(
        self,
        camera_id: str,
        zones: list[ZoneRule] | None = None,
        lines: list[LineRule] | None = None,
        near_miss_distance: float = 70.0,
        near_miss_cooldown: float = 3.0,
    ) -> None:
        self.camera_id = camera_id
        self.zones = zones or []
        self.lines = lines or []
        self.near_miss_distance = near_miss_distance
        self.near_miss_cooldown = near_miss_cooldown
        self.zone_state: dict[tuple[int, str], bool] = {}
        self.zone_entered_at: dict[tuple[int, str], float] = {}
        self.dwell_emitted: set[tuple[int, str]] = set()
        self.line_side: dict[tuple[int, str], float] = {}
        self.near_miss_at: dict[tuple[int, int], float] = {}
        self.class_counts: dict[str, int] = defaultdict(int)

    def process(self, tracks: list[Track]) -> list[AnalyticsEvent]:
        events: list[AnalyticsEvent] = []
        now = monotonic()
        active_ids = {track.track_id for track in tracks}
        self.class_counts = defaultdict(int)
        for track in tracks:
            self.class_counts[track.detection.class_name] += 1
            events.extend(self._zones(track, now))
            events.extend(self._lines(track))
        events.extend(self._near_misses(tracks, now))
        self._cleanup(active_ids)
        return events

    def _zones(self, track: Track, now: float) -> list[AnalyticsEvent]:
        result: list[AnalyticsEvent] = []
        for zone in self.zones:
            if zone.classes and track.detection.class_name not in zone.classes:
                continue
            key = (track.track_id, zone.name)
            inside = point_in_polygon(track.center, zone.points)
            previous = self.zone_state.get(key, False)
            if inside and not previous:
                self.zone_entered_at[key] = now
                result.append(self._event(EventType.ZONE_ENTER, track, f"{track.detection.class_name} #{track.track_id} entered {zone.name}", {"zone": zone.name}))
            elif previous and not inside:
                entered = self.zone_entered_at.pop(key, now)
                self.dwell_emitted.discard(key)
                result.append(self._event(EventType.ZONE_EXIT, track, f"{track.detection.class_name} #{track.track_id} left {zone.name}", {"zone": zone.name, "dwell_seconds": round(now - entered, 2)}))
            if inside and zone.dwell_seconds > 0 and key not in self.dwell_emitted:
                entered = self.zone_entered_at.get(key, now)
                if now - entered >= zone.dwell_seconds:
                    self.dwell_emitted.add(key)
                    result.append(self._event(EventType.DWELL, track, f"{track.detection.class_name} #{track.track_id} exceeded dwell time in {zone.name}", {"zone": zone.name, "threshold_seconds": zone.dwell_seconds}))
            self.zone_state[key] = inside
        return result

    def _lines(self, track: Track) -> list[AnalyticsEvent]:
        result: list[AnalyticsEvent] = []
        for line in self.lines:
            if line.classes and track.detection.class_name not in line.classes:
                continue
            key = (track.track_id, line.name)
            current = side_of_line(track.center, line.start, line.end)
            previous = self.line_side.get(key)
            if previous is not None and current != 0 and previous != 0 and (current > 0) != (previous > 0):
                result.append(self._event(EventType.LINE_CROSS, track, f"{track.detection.class_name} #{track.track_id} crossed {line.name}", {"line": line.name}))
            self.line_side[key] = current
        return result

    def _near_misses(self, tracks: list[Track], now: float) -> list[AnalyticsEvent]:
        result: list[AnalyticsEvent] = []
        for i, first in enumerate(tracks):
            for second in tracks[i + 1 :]:
                classes = {first.detection.class_name, second.detection.class_name}
                if not ({"person", "forklift"} <= classes):
                    continue
                distance = hypot(first.center[0] - second.center[0], first.center[1] - second.center[1])
                if distance > self.near_miss_distance:
                    continue
                pair = tuple(sorted((first.track_id, second.track_id)))
                last = self.near_miss_at.get(pair, -1e9)
                if now - last < self.near_miss_cooldown:
                    continue
                if not self._approaching(first, second):
                    continue
                self.near_miss_at[pair] = now
                result.append(
                    AnalyticsEvent(
                        EventType.NEAR_MISS,
                        self.camera_id,
                        first.track_id,
                        first.detection.class_name,
                        f"Near-miss between #{first.track_id} and #{second.track_id}",
                        {"track_ids": list(pair), "distance_px": round(distance, 1)},
                    )
                )
        return result

    @staticmethod
    def _approaching(first: Track, second: Track) -> bool:
        if len(first.history) < 2 or len(second.history) < 2:
            return True
        prev = hypot(first.history[-2][0] - second.history[-2][0], first.history[-2][1] - second.history[-2][1])
        current = hypot(first.center[0] - second.center[0], first.center[1] - second.center[1])
        return current <= prev

    def _cleanup(self, active_ids: set[int]) -> None:
        for mapping in (self.zone_state, self.zone_entered_at, self.line_side):
            for key in list(mapping):
                if key[0] not in active_ids:
                    mapping.pop(key, None)
        self.dwell_emitted = {key for key in self.dwell_emitted if key[0] in active_ids}

    def _event(self, event_type: EventType, track: Track, message: str, metadata: dict) -> AnalyticsEvent:
        return AnalyticsEvent(event_type, self.camera_id, track.track_id, track.detection.class_name, message, metadata)
