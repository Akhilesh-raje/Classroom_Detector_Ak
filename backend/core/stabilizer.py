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

        # Per-row Y filter: when scaffold has _row metadata, a detection
        # can only match seats in its own row band.
        # Compute the Y midpoint between consecutive rows and use as hard boundary.
        row_ys = {}
        for seat in self.seats:
            r = seat.get("_row")
            if r is not None:
                row_ys.setdefault(r, []).append(seat["center"][1])
        # sorted list of (row_idx, mean_y)
        row_means = sorted(
            [(r, sum(v)/len(v)) for r, v in row_ys.items()],
            key=lambda x: x[1]
        )
        # Build Y boundaries: midpoint between consecutive rows
        row_y_bounds = {}   # row_idx -> (y_min, y_max)
        for i, (ridx, ry) in enumerate(row_means):
            y_lo = row_means[i-1][1] + (ry - row_means[i-1][1])/2 if i > 0 else 0
            y_hi = ry + (row_means[i+1][1] - ry)/2 if i < len(row_means)-1 else float("inf")
            row_y_bounds[ridx] = (y_lo, y_hi)

        # ── Step 1: match detections to existing seats ────────────────────────
        dets_sorted = sorted(dets, key=lambda d: (d["bbox"][0] + d["bbox"][2]) / 2)

        pairs = []
        for d_idx, det in enumerate(dets_sorted):
            cx, cy = _center(det["bbox"])
            for s_idx, seat in enumerate(self.seats):
                # Hard Y-band filter when row metadata is available
                if row_y_bounds:
                    r = seat.get("_row")
                    if r is not None and r in row_y_bounds:
                        y_lo, y_hi = row_y_bounds[r]
                        if not (y_lo <= cy <= y_hi):
                            continue
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
                # Population locked — assign to nearest unoccupied seat
                # that is in the same Y-band as the detection.
                free = [i for i in range(len(self.seats)) if i not in assigned_seats]

                # Apply Y-band filter to free seats too
                if row_y_bounds:
                    free_in_band = [
                        i for i in free
                        if self.seats[i].get("_row") is None or
                           self.seats[i].get("_row") not in row_y_bounds or
                           row_y_bounds[self.seats[i]["_row"]][0] <= cy <=
                           row_y_bounds[self.seats[i]["_row"]][1]
                    ]
                    # If no in-band free seat, discard rather than cross-row assign.
                    # This detection is either a duplicate or a new person beyond
                    # the calibrated seat count — don't corrupt another row's topology.
                    if not free_in_band:
                        assigned_dets.add(d_idx)   # mark consumed, no seat assigned
                        continue
                    free = free_in_band

                target_idx = (
                    min(free, key=lambda i: math.hypot(
                        cx - self.seats[i]["center"][0],
                        cy - self.seats[i]["center"][1]))
                    if free else
                    min(range(len(self.seats)), key=lambda i: math.hypot(
                        cx - self.seats[i]["center"][0],
                        cy - self.seats[i]["center"][1]))
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
        """
        Sort seats row-by-row (top-to-bottom), then left-to-right within each row.
        Row structure is discovered from the EMA-stable seat centers — no count
        is assumed.  row_layout is recomputed on each gridCell query so it
        benefits from continued EMA convergence after the lock.
        """
        if not self.seats:
            return

        rows = _discover_rows(self.seats)

        # Sort rows top-to-bottom, seats within each row left-to-right
        rows_sorted = sorted(rows, key=lambda r: sum(s["center"][1] for s in r) / len(r))
        flat = []
        for row in rows_sorted:
            flat.extend(sorted(row, key=lambda s: s["center"][0]))

        for i, seat in enumerate(flat):
            seat["id"] = i + 1

        self.seats = flat
        self.seats_sorted = True
        # row_layout intentionally not cached here — recomputed lazily
        # by _row_layout property so later EMA convergence is captured.

    @property
    def row_layout(self) -> list:
        """
        Discover and return row sizes from current EMA-stable seat centers.
        Recomputed on every access so it improves as EMA converges post-lock.
        Only meaningful after seats_sorted is True.
        """
        if not self.seats_sorted:
            return []
        rows = _discover_rows(self.seats)
        rows_sorted = sorted(rows, key=lambda r: sum(s["center"][1] for s in r) / len(r))
        return [len(r) for r in rows_sorted]


# ── Module-level helpers ──────────────────────────────────────────────────────

def _discover_rows(seats: list) -> list:
    """
    Discover row structure from EMA-stable seat centers using ratio-gap
    partitioning on raw Y.

    Called only after the stabilizer has locked (EMA-converged positions),
    so the seat centers are stable and the inter-row gap reliably dominates.

    Algorithm:
      1. Sort seats by raw Y.
      2. Compute consecutive gaps.
      3. A gap is a row boundary only if it is at least GAP_RATIO times
         larger than the next-largest gap.  This makes the threshold
         data-driven and handles classrooms where within-row depth variation
         produces gaps almost as large as the row separation.
      4. Collect all such boundaries (supports 2-row, 3-row, etc).

    ponytail: O(n log n) sort + O(n) scan — fine for n ≤ 20.
    """
    if not seats:
        return []
    if len(seats) == 1:
        return [seats]

    sorted_seats = sorted(seats, key=lambda s: s["center"][1])
    ys = [s["center"][1] for s in sorted_seats]
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]

    if not gaps:
        return [sorted_seats]

    # Sort gaps descending to find the dominant ones
    sorted_gaps = sorted(gaps, reverse=True)
    largest = sorted_gaps[0]
    second  = sorted_gaps[1] if len(sorted_gaps) > 1 else 0

    # A gap qualifies as a row boundary only if it is substantially larger
    # than the next candidate.  Ratio of 2.0 means the inter-row gap must
    # be at least twice the within-row scatter maximum.
    # ponytail: fixed ratio; upgrade to Otsu thresholding on gap histogram
    #           if classrooms with more than 3 rows are needed.
    GAP_RATIO = 2.0
    threshold = max(largest / GAP_RATIO, second * GAP_RATIO)

    rows = [[sorted_seats[0]]]
    for i, gap in enumerate(gaps):
        if gap >= threshold and gap == largest:
            # Only the dominant gap(s) split rows
            rows.append([])
        rows[-1].append(sorted_seats[i + 1])

    # If threshold logic produced only one row (no gap qualified),
    # fall back to splitting at every gap that is >= largest/GAP_RATIO
    if len(rows) == 1 and len(seats) > 3:
        rows = [[sorted_seats[0]]]
        for i, gap in enumerate(gaps):
            if gap >= largest / GAP_RATIO:
                rows.append([])
            rows[-1].append(sorted_seats[i + 1])

    return rows

def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _update_seat(seat: dict, bbox: list, ts: float):
    """EMA-update the seat centre and record the latest bbox."""
    cx, cy = _center(bbox)
    sx, sy = seat["center"]
    # Scaffold seats (_row present) use very slow EMA to preserve physical anchor.
    # Discovery-mode seats use normal EMA α=0.12.
    alpha = 0.02 if "_row" in seat else 0.12
    seat["center"]    = (sx*(1-alpha) + cx*alpha, sy*(1-alpha) + cy*alpha)
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
import importlib
_mod = importlib.import_module("core.spatial.stabilizer")
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
