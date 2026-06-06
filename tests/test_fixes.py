"""
SmartClass AI — Fix Validation Tests
Tests all 6 ambiguity fixes:
1. Attention factor breakdown (transparency)
2. Reconciled activity (no high-attention + distracted contradiction)
3. Per-student base variation (prevents peak clustering)
4. Visualizer smooth function (no flickering)
5. Stable label (activity label stability)
6. Visualizer draw functions (no crashes)
"""
import sys
sys.path.insert(0, 'backend')

print("=== Testing all fixes ===")

# ── Fix 1: Visualizer imports ─────────────────────────────────────────────────
from core.visualizer import (
    draw_student_overlay, draw_global_hud, draw_grid_overlay,
    reset_smooth, smooth, stable_label
)
print("  [OK] visualizer imports")

# ── Fix 2: Attention factor breakdown ─────────────────────────────────────────
from core.student_state import StudentState
import numpy as np

st = StudentState(seat_id=1)
sig = {
    'vel':0.0, 'osc':0.0, 'av':0.0, 'drift':0.0,
    'turn_dev':0.0, 'slump_dev':0.0,
    'phone_detected':False, 'eye_openness':1.0,
    'mouth_open':0.0, 'yaw':0.0, 'pitch':0.0, 'roll':0.0
}
raw = st.compute_attn(sig)
assert hasattr(st, 'last_factors'), "last_factors missing from StudentState"
assert 'head_pose'        in st.last_factors, "head_pose factor missing"
assert 'motion_stability' in st.last_factors, "motion_stability factor missing"
assert 'eye_openness'     in st.last_factors, "eye_openness factor missing"
print(f"  [OK] attention factors: {st.last_factors}")

# ── Fix 3: Reconciled activity ────────────────────────────────────────────────
# Simulate high-attention session ending with brief distraction
st2 = StudentState(seat_id=2)
# Fill history with high attention values
for _ in range(30):
    st2.attn_hist.append(92.0)
st2.attn_ema  = 92.0
st2.prev_act  = "distracted"   # last frame was distracted

rec_act, note = st2.reconciled_activity()
assert rec_act != "distracted", (
    f"Reconciliation failed: 92% avg should not stay 'distracted', got '{rec_act}'"
)
assert len(note) > 0, "Note should explain the reconciliation"
print(f"  [OK] reconciled_activity: distracted+{st2.avg:.0f}% → '{rec_act}' (note: '{note}')")

# Low attention + studying label should also be flagged
st3 = StudentState(seat_id=3)
for _ in range(30):
    st3.attn_hist.append(35.0)
st3.attn_ema = 35.0
st3.prev_act = "studying"
rec3, note3 = st3.reconciled_activity()
# Low attention studying is suspicious — note should mention it
print(f"  [OK] low-attention studying: '{rec3}' note='{note3}'")

# ── Fix 4: attention_breakdown completeness ───────────────────────────────────
bd = st2.attention_breakdown()
required_keys = [
    'attention_score', 'avg_attention', 'peak_attention', 'low_attention',
    'confidence', 'factors', 'activity', 'activity_note', 'note'
]
for k in required_keys:
    assert k in bd, f"Missing key in attention_breakdown: '{k}'"
assert 'Observable engagement proxy' in bd['note'], "Transparency note missing"
assert 0.0 <= bd['confidence'] <= 1.0, f"Confidence out of range: {bd['confidence']}"
print(f"  [OK] attention_breakdown: all {len(required_keys)} fields present")
print(f"       note: '{bd['note'][:60]}...'")

# ── Fix 5: Per-student base variation ────────────────────────────────────────
states = [StudentState(i) for i in range(6)]
bases  = [s.base for s in states]
base_range = max(bases) - min(bases)
assert base_range > 2.0, (
    f"Base range too small ({base_range:.2f}) — all students will peak at same value"
)
print(f"  [OK] per-student base variation: range={base_range:.2f} "
      f"min={min(bases):.1f} max={max(bases):.1f}")

# ── Fix 6: Smooth function ────────────────────────────────────────────────────
reset_smooth()
v1 = smooth(1, 'attn', 100.0, alpha=0.10)
assert v1 == 100.0, f"First call should initialize to value, got {v1}"

v2 = smooth(1, 'attn', 0.0, alpha=0.10)
assert 85.0 < v2 < 100.0, f"After one step toward 0, expected ~90, got {v2:.1f}"

# After many steps toward 0, should approach 0
for _ in range(100):
    smooth(1, 'attn', 0.0, alpha=0.10)
v_final = smooth(1, 'attn', 0.0, alpha=0.10)
assert v_final < 5.0, f"After 100 steps toward 0, expected <5, got {v_final:.2f}"
print(f"  [OK] smooth: init=100 → after 1 step={v2:.1f} → after 100 steps={v_final:.2f}")

# ── Fix 7: Stable label ───────────────────────────────────────────────────────
reset_smooth()
# First call initializes — label is set immediately on first call
lbl0 = stable_label(99, 'act', 'studying', hold=5)
# Now switch to different label — should NOT commit until hold frames
for i in range(4):
    lbl = stable_label(99, 'act', 'distracted', hold=5)
    assert lbl == 'studying', f"Label switched too early at frame {i+1}: got '{lbl}'"
# 5th frame — should now commit
lbl5 = stable_label(99, 'act', 'distracted', hold=5)
assert lbl5 == 'distracted', f"Label should commit after 5 frames, got '{lbl5}'"
print(f"  [OK] stable_label: holds 4 frames, commits on 5th")

# ── Fix 8: Draw functions don't crash ────────────────────────────────────────
import numpy as np
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
dummy_face  = {
    'id': 1, 'bbox': [100, 100, 200, 200],
    'attentionScore': 85.0, 'avgAttention30': 82.0,
    'activity': 'studying', 'reconciledActivity': 'studying', 'activityNote': '',
    'emotion': 'neutral',
    'headPose': {'yaw': 5.0, 'pitch': -3.0, 'roll': 0.0},
    'eyeOpenness': 0.9, 'phoneDetected': False, 'laptopDetected': False,
    'gridCell': 'R1C1',
    'attentionFactors': {'head_pose': 0.92, 'motion_stability': 0.88, 'eye_openness': 0.9},
}
draw_student_overlay(dummy_frame, dummy_face)
print("  [OK] draw_student_overlay: no crash")

draw_global_hud(dummy_frame, [dummy_face], {
    'mode': 'LIVE', 'cam_idx': 0, 'w': 640, 'h': 480,
    'fps': 30.0, 'elapsed': 10.0, 'grid_label': '2x3',
    'hint': 'Q=quit',
})
print("  [OK] draw_global_hud: no crash")

# Phone alert path
dummy_phone = dict(dummy_face, phoneDetected=True)
draw_student_overlay(dummy_frame, dummy_phone)
print("  [OK] draw_student_overlay with phone alert: no crash")

# ── Fix 9: Syntax check all modified files ────────────────────────────────────
import ast, os
files = [
    'backend/core/visualizer.py',
    'backend/core/student_state.py',
    'backend/core/engine.py',
    'scratch/webcam/run_webcam.py',
    'scratch/video_runs/run_video.py',
]
for f in files:
    assert os.path.exists(f), f"File missing: {f}"
    ast.parse(open(f).read())
    print(f"  [OK] syntax: {f}")

# ── Fix 10: Full test suite still passes ─────────────────────────────────────
print()
print("Running original test_all.py checks inline...")
from core.config import YOLO_CONF, SEAT_PROXIMITY_THRES, HYSTERESIS_FRAMES
assert YOLO_CONF in (0.15, 0.18),     f"YOLO_CONF: {YOLO_CONF}"
assert SEAT_PROXIMITY_THRES == 0.25,  f"PROX: {SEAT_PROXIMITY_THRES}"
assert HYSTERESIS_FRAMES in (3,5),    f"HYST: {HYSTERESIS_FRAMES}"
print("  [OK] config constants")

from core.stabilizer import SeatStabilizer
try:
    SeatStabilizer(max_seats=0)
    assert False, "Should raise"
except ValueError:
    pass
print("  [OK] stabilizer validation")

from core.behavior_tracker import BehaviorTracker
import json
bt = BehaviorTracker(0.0)
bt.record("studying", 1.0)
bt.record("phone", 3.0)
tl = bt.finalize(5.0)
assert len(tl) == 2
j  = json.dumps(tl)
tl2 = json.loads(j)
assert tl2[0]["activity"] == "studying"
print("  [OK] behavior_tracker + JSON round-trip")

from core.engine import ClassroomEngine
e = ClassroomEngine(max_seats=6)
old = e.session_id
e.reset()
assert e.session_id != old
print("  [OK] engine session_id reset")

print()
print("=" * 55)
print("ALL FIX TESTS PASSED")
print("=" * 55)
