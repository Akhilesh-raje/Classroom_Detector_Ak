"""
SmartClass AI — Seat Stabilizer
Locks YOLO detections onto stable spatial seat positions with ghost-hold.

Key behaviours:
  1. Once a seat is discovered, its ID is permanent for the session.
  2. If a person is not detected for up to GHOST_HOLD_FRAMES frames, the seat
     is still returned with its last known bbox (ghost hold) — the count never
     drops for momentary occlusion or missed YOLO detections.
  3. After max_seats have been discovered, no new seat IDs are ever created —
     an unmatched detection is assigned to the nearest existing seat instead
     of spawning a ghost seat.
  4. Seat IDs are sorted left-to-right once the population stabilises.
"""
import time
import math
from core.config import MAX_SEATS, SEAT_PROXIMITY_THRES

# How many consecutive frames a seat can go unseen before it stops being
# reported (prevents truly gone students from lingering forever).
GHOST_HOLD_FRAMES = 8


class SeatStabilizer:
    def __init__(self, max_seats=MAX_SEATS):
        if not (1 <= max_seats <= 20):
            raise ValueError(f"max_seats must be in range [1, 20], got {max_seats}")
        self.max_seats = max_seats

        # Each seat dict:
        # {
        #   "id":          int,
        #   "center":      (cx, cy),        # EMA-smoothed position
        #   "last_seen":   float,           # time.time() of last actual detection
        #   "last_bbox":   [x1,y1,x2,y2],  # last known full-body bbox
        #   "missed":      int,             # consecutive frames NOT detected
        # }
        self.seats: list = []
        self.next_seat_id   = 1
        self.seats_sorted   = False

        # Once the population has been stable for STABLE_FRAMES consecutive
        # frames we lock the count and stop creating new seats.
        self._stable_count  = 0
        self._locked        = False
        STABLE_FRAMES       = 5   # frames at max_seats before locking

    # ── Public API ────────────────────────────────────────────────────────────

    def update_and_map(self, dets: list, W_frame: int, H_frame: int) -> list:
        """
        Map raw YOLO detections to stable persistent seat IDs.

        Returns a list of detection dicts, each augmented with "seat_id".
        Ghost seats (not detected this frame) are included with their last
        known bbox so the engine always sees a stable count.
        """
        current_time = time.time()
        diag         = math.hypot(W_frame, H_frame)
        threshold    = diag * SEAT_PROXIMITY_THRES

        # ── Step 1: match detections to existing seats ────────────────────────
        dets_sorted = sorted(dets, key=lambda d: (d["bbox"][0] + d["bbox"][2]) / 2)

        pairs = []
        for d_idx, det in enumerate(dets_sorted):
            cx, cy = _center(det["bbox"])
            for s_idx, seat in enumerate(self.seats):
                dist = math.hypot(cx - seat["center"][0], cy - seat["center"][1])
                if dist < threshold:
                    pairs.append((dist, d_idx, s_idx))

        pairs.sort(key=lambda x: x[0])

        assigned_dets  = set()
        assigned_seats = set()
        frame_results  = {}          # seat_id → augmented det

        for dist, d_idx, s_idx in pairs:
            if d_idx in assigned_dets or s_idx in assigned_seats:
                continue
            det  = dets_sorted[d_idx]
            seat = self.seats[s_idx]
            _update_seat(seat, det["bbox"], current_time)
            augmented = _augment(det, seat["id"])
            frame_results[seat["id"]] = augmented
            assigned_dets.add(d_idx)
            assigned_seats.add(s_idx)

        # ── Step 2: unmatched detections ──────────────────────────────────────
        for d_idx, det in enumerate(dets_sorted):
            if d_idx in assigned_dets:
                continue
            cx, cy = _center(det["bbox"])

            if self._locked:
                # Population locked — assign to nearest unoccupied seat,
                # or nearest seat overall (don't create new IDs ever).
                free = [i for i in range(len(self.seats)) if i not in assigned_seats]
                target_idx = (
                    min(free,      key=lambda i: math.hypot(cx - self.seats[i]["center"][0], cy - self.seats[i]["center"][1]))
                    if free else
                    min(range(len(self.seats)), key=lambda i: math.hypot(cx - self.seats[i]["center"][0], cy - self.seats[i]["center"][1]))
                )
                seat = self.seats[target_idx]
                _update_seat(seat, det["bbox"], current_time)
                augmented = _augment(det, seat["id"])
                frame_results[seat["id"]] = augmented
                assigned_dets.add(d_idx)
                assigned_seats.add(target_idx)

            else:
                # Still discovering — try free seats first, then create new
                free = [i for i in range(len(self.seats)) if i not in assigned_seats]
                if free:
                    target_idx = min(free, key=lambda i: math.hypot(cx - self.seats[i]["center"][0], cy - self.seats[i]["center"][1]))
                    seat = self.seats[target_idx]
                    _update_seat(seat, det["bbox"], current_time)
                    frame_results[seat["id"]] = _augment(det, seat["id"])
                    assigned_dets.add(d_idx)
                    assigned_seats.add(target_idx)
                elif len(self.seats) < self.max_seats:
                    new_seat = {
                        "id":        self.next_seat_id,
                        "center":    (cx, cy),
                        "last_seen": current_time,
                        "last_bbox": list(det["bbox"]),
                        "missed":    0,
                    }
                    self.seats.append(new_seat)
                    self.next_seat_id += 1
                    frame_results[new_seat["id"]] = _augment(det, new_seat["id"])
                    assigned_dets.add(d_idx)
                    assigned_seats.add(len(self.seats) - 1)
                # else: truly extra detection beyond max_seats — discard

        # ── Step 3: update missed counters for unmatched seats ────────────────
        matched_seat_indices = set(assigned_seats)
        for s_idx, seat in enumerate(self.seats):
            if s_idx not in matched_seat_indices:
                seat["missed"] += 1
            else:
                seat["missed"] = 0

        # ── Step 4: ghost hold — re-insert seats not seen this frame ─────────
        # Only re-insert seats within the ghost window.
        for seat in self.seats:
            if seat["id"] not in frame_results and seat["missed"] <= GHOST_HOLD_FRAMES:
                # Re-inject with last known bbox so the overlay keeps showing
                ghost_det = {
                    "bbox":        seat["last_bbox"],
                    "person_bbox": seat["last_bbox"],
                    "confidence":  0.0,   # marks as ghost
                    "id":          None,
                    "crops":       [],
                    "ghost":       True,
                }
                frame_results[seat["id"]] = _augment(ghost_det, seat["id"])

        # ── Step 5: sort IDs once population is stable ────────────────────────
        if not self.seats_sorted:
            n = len(self.seats)
            if n == self.max_seats:
                self._stable_count += 1
                if self._stable_count >= 5:
                    self._sort_ids()
                    self._locked = True
            else:
                self._stable_count = 0

        # Re-apply sorted IDs to results if sort just happened
        if self.seats_sorted:
            # Build seat_id lookup by internal list position
            for result in frame_results.values():
                orig_id = result.get("seat_id")
                if orig_id is None:
                    continue
                # Find seat with this id and update in case renaming happened
                for seat in self.seats:
                    if seat["id"] == orig_id:
                        result["seat_id"] = seat["id"]
                        break

        return list(frame_results.values())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _sort_ids(self):
        """Sort seats left-to-right and renumber 1..N."""
        self.seats = sorted(self.seats, key=lambda s: s["center"][0])
        for i, seat in enumerate(self.seats):
            seat["id"] = i + 1
        self.seats_sorted = True


# ── Module-level helpers ──────────────────────────────────────────────────────

def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _update_seat(seat: dict, bbox: list, ts: float):
    """EMA-update the seat centre and record the latest bbox."""
    cx, cy = _center(bbox)
    sx, sy = seat["center"]
    seat["center"]    = (sx * 0.88 + cx * 0.12, sy * 0.88 + cy * 0.12)
    seat["last_seen"] = ts
    seat["last_bbox"] = list(bbox)
    seat["missed"]    = 0


def _augment(det: dict, seat_id: int) -> dict:
    """Return a copy of det with seat_id injected and person_bbox normalised."""
    d = det.copy()
    d["seat_id"] = seat_id
    # Ensure person_bbox always exists (some dets may lack it)
    if "person_bbox" not in d or not d["person_bbox"]:
        d["person_bbox"] = d["bbox"]
    return d
