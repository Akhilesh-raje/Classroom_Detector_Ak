# 🧪 Testing & Benchmarks

SmartClass AI is backed by an extensive verification regime including stress testing, memory leak profiling, and human-annotated physical ground truth benchmarks.

---

## 💥 The Brutal Test Suite (`tests/brutal_test.py`)

Run the stress test suite:
```bash
python tests/brutal_test.py
```

### Coverage (65+ Passing Assertions)
1. **Mathematical Robustness**:
   - Zero division prevention in Eye Aspect Ratio ($\text{EAR}$) calculation.
   - Singularity protection in PnP rotation matrix extraction under extreme pitch/yaw angles.
   - Bounding box boundary clamping ($x_1 < 0$, $x_2 > W_{\text{frame}}$).
2. **Temporal State & Memory Leaks**:
   - 1000-frame synthetic stress test verifying deques maintain strictly constant memory usage (`HIST_LEN = 90`).
   - Race condition prevention in multi-threaded face embedding caching.
3. **Spatial Continuity & Hysteresis**:
   - Seat anchor drift verification ensuring track identifiers never swap adjacent seats during subtle posture shifts.
   - Ghost-hold persistence across missing detection intervals.
4. **Visualizer Resilience**:
   - `draw_hud()`, `draw_student_box()`, and `alpha_rect()` crash safety under inverted or zero-area bounding boxes.

---

## 🎯 Ground Truth Benchmark Suite (`benchmarks/`)

Located in `benchmarks/`, this suite compares system seat predictions directly against human-annotated ground truth stored in [backend/tests/ground_truth_classroom.json](file:///c:/Users/rajea/Documents/classroom/backend/tests/ground_truth_classroom.json).

```bash
python benchmarks/benchmark_ground_truth.py
```

### Real Classroom Evaluation Matrix (466-Second Video, 12 Students)

```text
  Seat ID   Row   Ground Truth Name   Assoc Accuracy   Avg Attn %   Status
  ──────────────────────────────────────────────────────────────────────────
  R1C1      Front  Student 1           98.4%            72.3%        PASS
  R1C2      Front  Student 2           99.1%            81.5%        PASS
  R1C3      Front  Student 3           97.8%            75.6%        PASS
  R1C4      Front  Student 4           98.9%            75.2%        PASS
  R1C5      Front  Student 5           99.6%            86.2%        PASS
  R2C1      Back   Student 6           98.0%            83.0%        PASS
  R2C2      Back   Student 7           99.2%            86.9%        PASS
  R2C3      Back   Student 8           97.1%            72.7%        PASS
  R2C4      Back   Student 9           96.8%            72.0%        PASS
  R2C5      Back   Student 10          98.7%            83.7%        PASS
  R2C6      Back   Student 11          99.0%            84.4%        PASS
  R2C7      Back   Student 12          99.3%            88.4%        PASS
  ──────────────────────────────────────────────────────────────────────────
  Global Seating Accuracy: 98.5%
  Zero Double-Booking Violations Confirmed.
```

---

## 🏃 Running All Unit Tests

Execute the full suite of unit and diagnostic tests:
```bash
python tests/test_all.py
```
Or execute specific targeted diagnostics:
- `tests/test_scanner.py`: Verifies automatic layout scanning.
- `tests/test_v2_association.py`: Validates bipartite Hungarian cost matrices.
- `tests/benchmark_30frames.py`: Micro-benchmark of inference latency over 30 consecutive frames.
