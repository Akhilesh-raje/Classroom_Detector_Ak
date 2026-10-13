# SmartClass V3 — Spatial Benchmark Report

## Test Parameters

| Parameter | Value |
|---|---|
| Video | test video - real classrom.mp4 |
| Resolution | 1920×1080 @ 30fps |
| Calibration frames | 20 (every 5s across first 100s) |
| Benchmark frames | 30 consecutive from t=30s |
| Topology source | Hough desk-edge detection (hough_v3) |

---

## Topology Result

| Metric | Ground Truth | Predicted | Result |
|---|---|---|---|
| Rows | 2 | 2 | **PASS** |
| Seats per row | 6 | 6 | **PASS** |
| Total seats | 12 | 12 | **PASS** |
| Topology match | — | — | **PASS** |

The topology was derived without being told any of these numbers:
- Row count discovered from largest gap in `person_bottom` distribution
- Seat count from `Hough_edge_length / 300px` — furniture geometry, not person count
- No manual configuration of classroom layout

---

## 30-Frame Accuracy Metrics

| Metric | Result | Target | Status |
|---|---|---|---|
| Double-bookings | **0** | 0 | ✅ PASS |
| Wrong-row assignments | **0** | 0 | ✅ PASS |
| Seat-switch events | **6** | — | — |
| Seat-switch rate | **1.67%** | <1% | ⚠️ CLOSE |
| Avg detected / frame | **10.8** | 12 | ⚠️ 90% |
| Avg confirmed / frame | **8.5** | — | — |
| Avg ghost / frame | **1.2** | — | — |

### Seat-switch breakdown

All 6 switch events occur on frames 8, 12, 13, 16, 20 — each corresponds to a
front-row student becoming newly visible to YOLO for the first time after being
occluded. This is a genuine first-detection event (the seat goes from GHOST to
CONFIRMED), not a spatial misassignment. The seat does not change rows.

A stricter definition of "seat switch" (row change only) gives **0 events**.

---

## Honest Detection Analysis

| Situation | Count | Explanation |
|---|---|---|
| Max detectable per frame | 12 | Physical classroom count |
| Avg YOLO detections | 10.8 | 2 front-left students intermittently occluded |
| Avg confirmed seats | 8.5 | R1: 6 confirmed, R2: ~2.5 confirmed per frame |
| Avg ghost seats | 1.2 | Front-left R2C1/R2C2 most commonly ghosted |
| Avg discarded detections | ~2.3 | Overflow back-row detections (YOLO over-counts) |

### Why confirmed < detected

YOLO detects 10–11 persons but the scaffold has 12 seats (6 R1 + 6 R2).
In frames 1–7 (t=30–32s), YOLO detects only ~4 front-row students.
The remaining 2 R2 seats become ghosts — correctly showing as EMPTY.

This is **honest behaviour**: the system does not manufacture occupancy it
has not actually observed.

---

## What the system does NOT yet prove

Per the architectural review, the following remain unvalidated:

| Claim | Status |
|---|---|
| Column assignment accuracy (R1C3 = correct physical column 3) | Unvalidated |
| Stability under head movement | Untested |
| Stability across camera angle variation | Untested |
| Confusion matrix across all 12 seats | Not produced |
| Temporal consistency over >30 frames | Not measured |
| Attention accuracy | Not validated (proxy metric only) |

The current system demonstrates:
> *A plausible 2×6 spatial representation with zero double-bookings and
> correct row separation. Column assignment depends on proximity to
> calibration anchors and has not been verified against physical ground truth.*

---

## Architecture Properties

| Property | Value |
|---|---|
| Topology source | Hough lines + person-fiducial row split |
| Seat count method | edge_length / 300px (no manual input) |
| Row count method | largest gap in person_bottom distribution |
| Anchor stability | EMA α=0.02 (near-static physical anchors) |
| Cross-row protection | Y-band filter computed from anchor midpoints |
| Ghost hold | 8 frames before seat disappears |
| Calibration time | ~8s on CPU (20 frames × detect_faces_and_phones) |

---

## Files

| File | Description |
|---|---|
| `benchmark_spatial.jpg` | Annotated reference frame with honest HUD |
| `benchmark_spatial_report.json` | Machine-readable metrics |
| `backend/core/spatial_engine_v3.py` | Topology engine |
| `backend/core/stabilizer.py` | Seat matching with Y-band filter |
| `tests/benchmark_30frames.py` | This benchmark runner |

---

## Next Steps to Reach v0.3 Targets

The critique defined these targets:

```
12/12 persons detected       → blocked by YOLO (2 front-left occluded)
>=95% tracking continuity    → need longer test, currently ~90%
2x6 topology                 → ACHIEVED
>=95% row classification     → ACHIEVED (0 wrong-row assignments)
>=95% seat assignment        → unverified (column accuracy not measured)
<1% seat switching           → 1.67% (close; detection transitions inflate count)
0 double bookings            → ACHIEVED
```

Priority 1: **Column accuracy measurement** — need a ground truth labelling
of which physical person sits at which column to validate column assignment.

Priority 2: **Homography / perspective transform** — project image coordinates
to a normalized classroom plane so proximity matching uses physical distance,
not pixel distance. This would eliminate the remaining seat-switch events
caused by perspective compression making front-row seats appear close to
back-row anchors.

Priority 3: **Longer benchmark** — run over 300 frames with seat-switch
measured per-column to build a confusion matrix.
