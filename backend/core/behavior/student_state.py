"""
SmartClass AI — Per-Student Temporal State (MediaPipe Enhanced)
Keeps track of moving signals, EMA-smoothed attention, and activity state machines
incorporating head pose, eye openness, mouth opening, phone/laptop/earphone detection.
"""
import numpy as np
import math
from collections import deque
from core.config import EMA_A, HIST_LEN, HYSTERESIS_FRAMES


class StudentState:
    def __init__(self, seat_id: int):
        self.id          = seat_id
        self.attn_ema    = None
        self.attn_hist   = deque(maxlen=HIST_LEN)
        self.vel_hist    = deque(maxlen=20)
        self.area_hist   = deque(maxlen=15)
        self.center_hist = deque(maxlen=10)
        self.prev_act    = "neutral"
        self.base        = float(np.random.uniform(74, 96))

        # Temporal buffers for robust states
        self.drowsy_buf  = 0

        # Electronics frame counters
        self.phone_frame_count:    int = 0
        self.laptop_frame_count:   int = 0
        self.earphone_frame_count: int = 0

        # Hysteresis state machine
        self._act_candidate:       str = "neutral"
        self._act_candidate_count: int = 0

        # Attention factor breakdown (for transparency)
        self.last_factors: dict = {
            "head_pose": 1.0, "face_visibility": 1.0,
            "motion_stability": 1.0, "eye_openness": 1.0
        }

    def smooth(self, raw: float) -> float:
        if self.attn_ema is None:
            self.attn_ema = raw
        else:
            self.attn_ema = EMA_A * raw + (1 - EMA_A) * self.attn_ema
        self.attn_hist.append(self.attn_ema)
        return round(self.attn_ema, 1)

    @property
    def avg(self) -> float:
        return round(sum(self.attn_hist) / len(self.attn_hist), 1) if self.attn_hist else 0.0

    @property
    def peak(self) -> float:
        return round(max(self.attn_hist), 1) if self.attn_hist else 0.0

    @property
    def low(self) -> float:
        return round(min(self.attn_hist), 1) if self.attn_hist else 0.0

    def extract_signals(self, bbox, person_bbox, phone_detected=False,
                        eye_openness=1.0, mouth_open=0.0, head_pose=None) -> dict:
        x1, y1, x2, y2 = bbox
        cx, cy  = (x1 + x2) / 2, (y1 + y2) / 2
        area    = (x2 - x1) * (y2 - y1)
        diag    = math.hypot(x2 - x1, y2 - y1) + 1e-6

        px1, py1, px2, py2 = person_bbox
        pw = max(1, px2 - px1)
        ph = max(1, py2 - py1)

        head_rel_x = (cx - px1) / pw
        head_rel_y = (cy - py1) / ph

        turn_dev  = abs(head_rel_x - 0.50)
        slump_dev = max(0.0, head_rel_y - 0.20)

        vel = 0.0
        if self.center_hist:
            lx, ly = self.center_hist[-1]
            vel = math.hypot(cx - lx, cy - ly) / diag

        self.center_hist.append((cx, cy))
        self.vel_hist.append(vel)
        self.area_hist.append(area)

        osc = float(np.std(list(self.vel_hist)[-6:])) if len(self.vel_hist) >= 6 else 0.0

        av = (float(np.std(list(self.area_hist)[-6:]) /
               (np.mean(list(self.area_hist)[-6:]) + 1e-6))
              if len(self.area_hist) >= 6 else 0.0)

        ys = [pt[1] for pt in list(self.center_hist)[-5:]]
        drift = max(0, ys[-1] - ys[0]) / (y2 - y1 + 1e-6) if len(ys) >= 5 else 0.0

        if head_pose is None:
            head_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        return {
            "vel": vel,
            "osc": osc,
            "av": av,
            "drift": drift,
            "turn_dev": turn_dev,
            "slump_dev": slump_dev,
            "phone_detected": phone_detected,
            "eye_openness": eye_openness,
            "mouth_open": mouth_open,
            "yaw": head_pose.get("yaw", 0.0),
            "pitch": head_pose.get("pitch", 0.0),
            "roll": head_pose.get("roll", 0.0)
        }

    def compute_attn(self, sig: dict) -> float:
        """
        Compute attention score 0-100 from observable engagement proxies.
        """
        if sig["phone_detected"]:
            self.last_factors = {"head_pose": 0.0, "face_visibility": 0.0,
                                  "motion_stability": 0.0, "eye_openness": 0.0}
            return float(np.random.uniform(5.0, 12.0))

        vel_score   = 1.0 - min(1.0, sig["vel"] / 0.06)
        osc_score   = 1.0 - min(1.0, sig["osc"] / 0.025)
        area_score  = 1.0 - min(1.0, sig["av"] / 0.10)
        drift_score = 1.0 - min(1.0, sig["drift"] / 0.15)

        yaw_dev   = abs(sig["yaw"])
        # Only penalise yaw beyond 35° (students naturally look slightly left/right)
        yaw_score = 1.0 - min(1.0, max(0.0, yaw_dev - 35.0) / 25.0)

        pitch_dev   = sig["pitch"]
        # Only penalise pitch beyond 25° (writing/reading causes natural downward pitch)
        pitch_score = 1.0 - min(1.0, max(0.0, abs(pitch_dev) - 25.0) / 25.0)

        head_pose_score = (yaw_score * 0.6 + pitch_score * 0.4)

        eye_score   = sig["eye_openness"]
        # Only penalise if eyes nearly fully closed (< 0.3), not normal squinting
        eye_penalty = max(0.0, eye_score / 0.3) if eye_score < 0.3 else 1.0

        motion_score = (vel_score * 0.35 + osc_score * 0.25 +
                        area_score * 0.25 + drift_score * 0.15)

        self.last_factors = {
            "head_pose":        round(head_pose_score, 3),
            "face_visibility":  round(1.0 - min(1.0, sig["turn_dev"] / 0.3), 3),
            "motion_stability": round(motion_score, 3),
            "eye_openness":     round(eye_score, 3),
        }

        raw = (
            motion_score    * 0.25 +
            head_pose_score * 0.60 +
            eye_score       * 0.15
        ) * 100.0 * eye_penalty

        base_weight = 0.18 * head_pose_score
        raw = raw * (1.0 - base_weight) + self.base * base_weight
        noise_scale = 0.6 + abs(self.base - 88) * 0.10
        raw += float(np.random.uniform(-noise_scale, noise_scale))
        return max(5.0, min(100.0, raw))

    def reconciled_activity(self) -> tuple[str, str]:
        """
        Reconcile last activity label against session average and peak.

        Real-world reality:
          - A student with avg 65% and peak 86% is NOT "distracted" —
            they are engaged but have natural variation (looking at board,
            writing, turning to a classmate briefly).
          - Only flag distracted if BOTH avg AND peak are low.
        """
        act  = self.prev_act
        note = ""

        # Strong reconciliation — high avg overrides momentary distracted label
        if self.avg >= 80 and act in ("distracted", "fidgeting"):
            note = f"briefly {act}"
            act  = "studying"
        elif self.avg >= 72 and act == "distracted":
            note = "momentarily distracted"
            act  = "attentive"
        # Peak-based reconciliation — high peak with moderate avg means engaged
        elif self.peak >= 78 and self.avg >= 60 and act == "distracted":
            note = "naturally varying attention"
            act  = "attentive"
        elif self.avg < 45 and act in ("studying", "attentive"):
            note = "low avg despite studying label"

        return act, note

    def attention_breakdown(self) -> dict:
        act, note = self.reconciled_activity()
        return {
            "attention_score":  self.attn_ema or 0.0,
            "avg_attention":    self.avg,
            "peak_attention":   self.peak,
            "low_attention":    self.low,
            "confidence":       round(min(1.0, len(self.attn_hist) / 30), 2),
            "factors":          self.last_factors,
            "activity":         act,
            "activity_note":    note,
            "note": (
                "Observable engagement proxy based on head pose, eye openness, "
                "and motion stability — not a measure of cognitive comprehension."
            ),
        }

    def classify_act(
        self,
        sig: dict,
        attn: float,
        laptop_detected: bool = False,
        earphone_detected: bool = False,
    ) -> str:
        v, o, av, dr = sig["vel"], sig["osc"], sig["av"], sig["drift"]
        yaw, pitch   = sig["yaw"], sig["pitch"]
        eye_open     = sig["eye_openness"]
        mouth_open   = sig["mouth_open"]

        if sig["phone_detected"]:
            self.phone_frame_count += 1
        if laptop_detected:
            self.laptop_frame_count += 1
        if earphone_detected:
            self.earphone_frame_count += 1

        # ── Priority-ordered classification ───────────────────────────────────
        # 1. Phone
        if sig["phone_detected"]:
            candidate = "phone"

        # 2. Drowsy — requires sustained near-closed eyes
        elif eye_open < 0.18:
            self.drowsy_buf += 1
            candidate = "drowsy" if self.drowsy_buf >= 5 else self.prev_act
        elif dr > 0.22 or (abs(pitch) > 35.0 and pitch < -20.0):
            self.drowsy_buf = max(0, self.drowsy_buf - 1)
            candidate = "drowsy"
        else:
            self.drowsy_buf = max(0, self.drowsy_buf - 1)

            # 3. Talking (mouth clearly open AND some head motion)
            if mouth_open > 0.30 and o > 0.015:
                candidate = "talking"

            # 4. Distracted — only flag if head is SIGNIFICANTLY turned AND
            #    attention score is also low (prevents false positives from
            #    students glancing at the board or writing)
            elif (abs(yaw) > 35.0 or pitch > 28.0 or pitch < -28.0) and attn < 65:
                candidate = "distracted"

            # 5. Laptop
            elif laptop_detected and abs(yaw) <= 35:
                candidate = "laptop"

            # 6. Studying — focused, low motion, good attention
            elif attn >= 72 and not laptop_detected and not earphone_detected and v < 0.04:
                candidate = "studying"

            # 7. Attentive — decent attention score
            elif attn >= 62:
                candidate = "attentive"

            # 8. Music
            elif earphone_detected and self.prev_act not in ("attentive", "studying"):
                candidate = "music"

            # 9. Fidgeting
            elif av > 0.095 and v < 0.05:
                candidate = "fidgeting"

            # 10. Kinematic distracted / neutral
            elif v > 0.07:
                candidate = "distracted"
            elif attn >= 40:
                candidate = "neutral"
            else:
                candidate = "distracted"

        # ── Hysteresis ────────────────────────────────────────────────────────
        if candidate == self._act_candidate:
            self._act_candidate_count += 1
        else:
            self._act_candidate       = candidate
            self._act_candidate_count = 1

        if self._act_candidate_count >= HYSTERESIS_FRAMES:
            self.prev_act = self._act_candidate

        return self.prev_act
