"""
SmartClass AI — Per-Student Behavior Tracker
Builds a chronological activity timeline for a single student during a session.
Each distinct activity run is recorded as an ActivitySegment.
"""
from dataclasses import dataclass, asdict


@dataclass
class ActivitySegment:
    activity:  str
    start_ts:  float   # seconds from session start
    end_ts:    float   # updated on next transition or finalize
    duration:  float   # end_ts - start_ts


class BehaviorTracker:
    """
    Records activity transitions for one student and produces a serialisable
    timeline at the end of a session.

    Usage:
        tracker = BehaviorTracker(session_start_time=time.time())
        # ... each processed frame:
        tracker.record(activity="studying", now_ts=time.time())
        # ... at session end:
        timeline = tracker.finalize(session_duration=elapsed_seconds)
    """

    def __init__(self, session_start_time: float) -> None:
        self._session_start: float = session_start_time
        self.timeline: list[ActivitySegment] = []
        self._current: dict | None = None

    def record(self, activity: str, now_ts: float) -> None:
        """
        Called once per processed frame (after upstream hysteresis has already
        committed the activity label).

        - If no segment is open, starts one.
        - If the activity is unchanged, the current segment continues silently.
        - If the activity changed, closes the current segment and opens a new one.
        """
        relative_ts = now_ts - self._session_start

        if self._current is None:
            # Start the very first segment
            self._current = {
                "activity": activity,
                "start_ts": relative_ts,
            }
            return

        if activity == self._current["activity"]:
            # Same activity — segment continues, nothing to do
            return

        # Activity changed — close the current segment
        end_ts = relative_ts
        duration = end_ts - self._current["start_ts"]
        segment = ActivitySegment(
            activity=self._current["activity"],
            start_ts=self._current["start_ts"],
            end_ts=end_ts,
            duration=duration,
        )
        self.timeline.append(segment)

        # Open a new segment for the new activity
        self._current = {
            "activity": activity,
            "start_ts": relative_ts,
        }

    def finalize(self, session_duration: float) -> list[dict]:
        """
        Closes the last open segment using *session_duration* as its end
        timestamp, then returns the full timeline as a list of plain dicts
        suitable for JSON serialisation.

        Args:
            session_duration: Total elapsed seconds for the session (used as
                              the end_ts of the final segment).

        Returns:
            List of dicts with keys: activity (str), start_ts (float),
            end_ts (float), duration (float).
        """
        if self._current is not None:
            end_ts = session_duration
            duration = end_ts - self._current["start_ts"]
            segment = ActivitySegment(
                activity=self._current["activity"],
                start_ts=self._current["start_ts"],
                end_ts=end_ts,
                duration=duration,
            )
            self.timeline.append(segment)
            self._current = None

        return [asdict(seg) for seg in self.timeline]
