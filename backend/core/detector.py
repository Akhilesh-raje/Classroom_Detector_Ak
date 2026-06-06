"""
SmartClass AI — YOLO & MediaPipe Detector & Crop Generator
Detects person bounding boxes, tracks persons & phones, crops face candidate regions,
and runs MediaPipe FaceLandmarker on face crops for detailed analysis.
"""
import cv2
import numpy as np
import logging
import os
import math
from core.config import YOLO_MODEL_PATH, YOLO_CONF, PERSIST

log = logging.getLogger("yolo_detector")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

# ── Model Path Resolution Helper ─────────────────────────────────────────────
def resolve_model_path(model_filename):
    """
    Dynamically resolve the absolute path to a model file.
    Checks inside the workspace-level models/ directory, relative path, and parent paths.
    """
    if os.path.isabs(model_filename) and os.path.exists(model_filename):
        return model_filename

    # Check 1: workspace root models/ folder (relative to this file backend/core/detector.py)
    core_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_models = os.path.normpath(os.path.join(core_dir, "..", "..", "models", model_filename))
    if os.path.exists(workspace_models):
        return workspace_models

    # Check 2: relative to current working directory (e.g., models/<filename>)
    cwd_models = os.path.normpath(os.path.join("models", model_filename))
    if os.path.exists(cwd_models):
        return os.path.abspath(cwd_models)

    # Check 3: parent models folder (e.g., ../models/<filename>)
    parent_models = os.path.normpath(os.path.join("..", "models", model_filename))
    if os.path.exists(parent_models):
        return os.path.abspath(parent_models)

    # Fallback to current working directory
    if os.path.exists(model_filename):
        return os.path.abspath(model_filename)

    # Fallback to direct backend folder path
    backend_fallback = os.path.normpath(os.path.join(core_dir, "..", model_filename))
    if os.path.exists(backend_fallback):
        return backend_fallback

    return model_filename


# ── YOLO Model Initialization ────────────────────────────────────────────────
_yolo = None

def _get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        model_path = resolve_model_path(YOLO_MODEL_PATH)
        _yolo = YOLO(model_path)
        log.info(f"YOLO detector model loaded: {model_path}")
    return _yolo


# ── MediaPipe Model Initialization ──────────────────────────────────────────
_mp_detector = None

def _get_mp_detector():
    global _mp_detector
    if _mp_detector is None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        model_path = resolve_model_path("face_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe face landmarker model task file not found at: {model_path}")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
            min_face_detection_confidence=0.1
        )
        _mp_detector = vision.FaceLandmarker.create_from_options(options)
        log.info(f"MediaPipe FaceLandmarker initialized from: {model_path}")
    return _mp_detector


# ── YOLO Detection & Tracking ────────────────────────────────────────────────
def track_persons(frame_bgr, min_conf: float = YOLO_CONF, persist=PERSIST):
    """Track persons (COCO class 0) with persistent IDs."""
    model = _get_yolo()
    results = model.track(frame_bgr, classes=[0], conf=min_conf, verbose=False, persist=persist)
    dets = []
    for r in results:
        if r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            for box, tid in zip(r.boxes, ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                dets.append({"bbox": [x1, y1, x2, y2], "confidence": float(box.conf[0]), "id": int(tid)})
        else:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                dets.append({"bbox": [x1, y1, x2, y2], "confidence": float(box.conf[0]), "id": None})
    return dets

def track_persons_and_phones(frame_bgr, min_conf: float = YOLO_CONF, persist=PERSIST):
    """Track persons (class 0), cell phones (class 67), and laptops (class 63)."""
    model = _get_yolo()
    # Track class 0 (person), class 67 (cell phone), and class 63 (laptop)
    results = model.track(frame_bgr, classes=[0, 67, 63], conf=min_conf, verbose=False, persist=persist)

    persons = []
    phones = []
    laptops = []

    for r in results:
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(r.boxes)
        for box, tid in zip(r.boxes, ids):
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            entry = {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "id": int(tid) if tid is not None else None,
            }
            if cls == 0:
                persons.append(entry)
            elif cls == 67:
                phones.append(entry)
            elif cls == 63:
                # Only accept laptop detections with high confidence AND meaningful size
                # (avoids misclassifying books, bags, or other rectangular objects)
                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h
                frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
                # Laptop must be: conf >= 0.50, area >= 1% of frame, wider than tall
                if conf >= 0.50 and box_area >= (frame_area * 0.01) and box_w > box_h:
                    laptops.append(entry)

    return persons, phones, laptops


# ── Crop Candidates Generator ────────────────────────────────────────────────
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _make_crop(person_bbox, frame_h, frame_w, head_frac, width_frac, pad_x_frac, pad_y_frac):
    """Build a single crop region from a person bbox."""
    x1, y1, x2, y2 = person_bbox
    pw, ph = x2 - x1, y2 - y1

    crop_h = int(ph * head_frac)
    cx     = (x1 + x2) // 2
    crop_w = int(pw * width_frac)

    fx1 = cx - crop_w // 2
    fy1 = y1
    fx2 = cx + crop_w // 2
    fy2 = y1 + crop_h

    # Apply padding
    px = int(crop_w * pad_x_frac)
    py = int(crop_h * pad_y_frac)
    fx1 = _clamp(fx1 - px, 0, frame_w)
    fy1 = _clamp(fy1 - py, 0, frame_h)
    fx2 = _clamp(fx2 + px, 0, frame_w)
    fy2 = _clamp(fy2 + py, 0, frame_h)

    if (fx2 - fx1) < 15 or (fy2 - fy1) < 15:
        return None
    return [fx1, fy1, fx2, fy2]

# Crop Profiles: (name, head_fraction, width_fraction, pad_x, pad_y)
_CROP_PROFILES = [
    ("tight",      0.35, 0.75, 0.20, 0.20),   # standard face crop
    ("wide_head",  0.50, 1.00, 0.30, 0.25),   # wider - catches profile heads
    ("upper_body", 0.65, 1.10, 0.35, 0.30),   # upper body - catches occluded heads
]

def person_to_crops(person_bbox, frame_h, frame_w) -> list:
    """Generate multiple crop candidates for one person detection."""
    crops = []
    for name, hf, wf, px, py in _CROP_PROFILES:
        box = _make_crop(person_bbox, frame_h, frame_w, hf, wf, px, py)
        if box:
            crops.append({"name": name, "bbox": box})
    return crops


# ── High-Level Face Detections API ───────────────────────────────────────────
def detect_faces(frame_bgr, min_conf: float = YOLO_CONF, persist=PERSIST):
    """
    Detect and track all persons, returning face candidate boxes.
    """
    h, w = frame_bgr.shape[:2]
    persons = track_persons(frame_bgr, min_conf=min_conf, persist=persist)

    faces = []
    for p in persons:
        crops = person_to_crops(p["bbox"], h, w)
        if not crops:
            continue
        faces.append({
            "id": p["id"],
            "bbox": crops[0]["bbox"],       # Tight crop as primary face box
            "crops": crops,
            "person_bbox": p["bbox"],
            "confidence": p["confidence"],
        })

    return faces

def _is_within_padded_bbox(inner_bbox, outer_bbox, pad_frac: float = 0.10) -> bool:
    """
    Check if the center of ``inner_bbox`` falls within ``outer_bbox`` expanded
    by ``pad_frac`` on each side.

    Args:
        inner_bbox: [x1, y1, x2, y2] of the inner box (e.g. laptop).
        outer_bbox: [x1, y1, x2, y2] of the outer box (e.g. person).
        pad_frac:   Fraction of each outer-bbox dimension to add as padding on
                    each side (default 0.10 = 10%).

    Returns:
        True if the center of inner_bbox is inside the padded outer_bbox.
    """
    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox

    # Center of the inner bbox
    cx = (ix1 + ix2) / 2.0
    cy = (iy1 + iy2) / 2.0

    # Padding amounts (10% of each outer dimension)
    pad_x = (ox2 - ox1) * pad_frac
    pad_y = (oy2 - oy1) * pad_frac

    return (ox1 - pad_x) <= cx <= (ox2 + pad_x) and (oy1 - pad_y) <= cy <= (oy2 + pad_y)


def detect_earphones_in_head_crop(frame_bgr, persons, min_conf: float = 0.20) -> list:
    """
    Earphone detection via head-crop YOLO pass.
    NOTE: yolo11s.pt is not trained for earphones/earbuds and produces too many
    false positives. This function returns an empty list until a dedicated
    earphone detection model is available.
    """
    return []


def detect_faces_and_phones(frame_bgr, min_conf: float = YOLO_CONF, persist=PERSIST):
    """
    Detect and track all persons, cell phones, laptops, and earphones.

    Returns:
        (faces, phones, laptops, earphones) — a 4-tuple where:
          - faces:     list of face-candidate dicts (id, bbox, crops, person_bbox, confidence)
          - phones:    list of phone dicts (bbox, confidence, id)
          - laptops:   list of laptop dicts (bbox, confidence, id)
          - earphones: list of earphone dicts (person_idx, bbox)
    """
    h, w = frame_bgr.shape[:2]
    persons, phones, laptops = track_persons_and_phones(frame_bgr, min_conf=min_conf, persist=persist)

    faces = []
    for p in persons:
        crops = person_to_crops(p["bbox"], h, w)
        if not crops:
            continue
        faces.append({
            "id": p["id"],
            "bbox": crops[0]["bbox"],       # Tight crop as primary face box
            "crops": crops,
            "person_bbox": p["bbox"],
            "confidence": p["confidence"],
        })

    earphones = detect_earphones_in_head_crop(frame_bgr, persons, min_conf=0.20)

    return faces, phones, laptops, earphones


# ── MediaPipe Crop Analysis Helpers ─────────────────────────────────────────
def estimate_head_pose(landmarks, w, h):
    """
    Estimate head pose (yaw, pitch, roll) from 3D FaceLandmarker landmarks
    using a 3D generic face model and cv2.solvePnP.
    """
    # 2D image points from landmarks
    # landmarks indices: 4=Nose, 152=Chin, 263=R Eye outer, 33=L Eye outer, 287=R Mouth corner, 57=L Mouth corner
    image_points = np.array([
        (landmarks[4].x * w, landmarks[4].y * h),       # Nose tip
        (landmarks[152].x * w, landmarks[152].y * h),   # Chin
        (landmarks[263].x * w, landmarks[263].y * h),   # Right eye outer corner
        (landmarks[33].x * w, landmarks[33].y * h),     # Left eye outer corner
        (landmarks[287].x * w, landmarks[287].y * h),   # Right mouth corner
        (landmarks[57].x * w, landmarks[57].y * h)      # Left mouth corner
    ], dtype="double")

    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Right eye outer corner
        (225.0, 170.0, -135.0),      # Left eye outer corner
        (-150.0, -150.0, -125.0),    # Right mouth outer corner
        (150.0, -150.0, -125.0)      # Left mouth outer corner
    ])

    # Approximate camera internals based on crop dimensions
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    dist_coeffs = np.zeros((4,1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        
    rmat, _ = cv2.Rodrigues(rotation_vector)
    
    # Calculate Euler angles
    sy = math.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(rmat[2,1] , rmat[2,2])
        y = math.atan2(-rmat[2,0], sy)
        z = math.atan2(rmat[1,0], rmat[0,0])
    else:
        x = math.atan2(-rmat[1,2], rmat[1,1])
        y = math.atan2(-rmat[2,0], sy)
        z = 0
        
    return {
        "yaw": round(math.degrees(y), 1),
        "pitch": round(math.degrees(x), 1),
        "roll": round(math.degrees(z), 1)
    }

def classify_emotion_from_blendshapes(bs):
    """
    Classify emotion from blendshape scores.
    """
    smile = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0
    frown = (bs.get("mouthFrownLeft", 0.0) + bs.get("mouthFrownRight", 0.0)) / 2.0
    brow_down = (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2.0
    brow_up = (bs.get("browOuterUpLeft", 0.0) + bs.get("browOuterUpRight", 0.0)) / 2.0
    jaw_open = bs.get("jawOpen", 0.0)
    
    if smile > 0.25:
        return "happy"
    elif brow_up > 0.30 and jaw_open > 0.25:
        return "surprised"
    elif brow_down > 0.35:
        return "angry"
    elif frown > 0.20 or bs.get("mouthStretchLeft", 0.0) > 0.20:
        return "sad"
    else:
        return "neutral"

def analyze_face_crop(crop):
    """
    Run MediaPipe FaceLandmarker on the BGR face crop.
    Returns: dict with head_pose, eye_openness, mouth_open, emotion or None if no face found.
    """
    if crop is None or crop.size == 0:
        return None
        
    try:
        import mediapipe as mp
        detector = _get_mp_detector()
        h, w = crop.shape[:2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        
        if not result.face_landmarks:
            return None
            
        lms = result.face_landmarks[0]
        
        # Extract blendshapes
        blendshapes_dict = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                blendshapes_dict[category.category_name] = category.score
                
        blink_l = blendshapes_dict.get("eyeBlinkLeft", 0.0)
        blink_r = blendshapes_dict.get("eyeBlinkRight", 0.0)
        eye_openness = float(1.0 - (blink_l + blink_r) / 2.0)
        
        mouth_open = float(blendshapes_dict.get("jawOpen", 0.0))
        head_pose = estimate_head_pose(lms, w, h)
        emotion = classify_emotion_from_blendshapes(blendshapes_dict)
        
        return {
            "success": True,
            "eye_openness": round(eye_openness, 3),
            "mouth_open": round(mouth_open, 3),
            "head_pose": head_pose,
            "emotion": emotion,
            "blendshapes": blendshapes_dict
        }
    except Exception as e:
        log.warning(f"Error in MediaPipe crop analysis: {e}")
        return None
