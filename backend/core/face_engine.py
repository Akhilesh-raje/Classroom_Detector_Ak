"""
SmartClass AI — YOLO Face Analysis Engine (with InsightFace recognition)
"""
import numpy as np
import cv2
from core.engine import ClassroomEngine
from core.face_recognition_engine import face_rec_engine

class FaceEngine:
    def __init__(self):
        self.engine = ClassroomEngine(max_seats=6)
        self.registered_encodings = []
        print("FaceEngine initialized with InsightFace recognition.")

    def process_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        engine_faces = self.engine.process_single_frame(frame_bgr, w, h, persist=True)

        faces = []
        for ef in engine_faces:
            # Extract real face embedding from the face crop
            x1, y1, x2, y2 = ef["bbox"]
            face_crop = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]

            embedding = None
            matched_id = None
            matched_name = "Unknown"
            matched_roll = ""
            match_conf = 0.0

            if face_crop.size > 0:
                embedding = face_rec_engine.extract_embedding(face_crop)
                if embedding is not None:
                    matched_id, matched_name, matched_roll, match_conf = face_rec_engine.match(embedding)

            faces.append({
                "bbox":               ef["bbox"],
                "head_pose":          ef["headPose"],
                "eye_openness":       ef.get("eyeOpenness", 1.0),
                "mouth_open":         ef.get("mouthOpen", 0.0),
                "phone_detected":     ef.get("phoneDetected", False),
                "laptop_detected":    ef.get("laptopDetected", False),
                "earphone_detected":  ef.get("earphoneDetected", False),
                "crop_used":          ef.get("cropUsed", "none"),
                "attention_score":    ef["attentionScore"],
                "activity":           ef["activity"],
                "emotion":            ef["emotion"],
                "encoding":           embedding if embedding is not None else np.zeros(512, dtype=np.float32),
                "matched_student_id": matched_id if matched_id else ef["id"],
                "matched_name":       matched_name,
                "matched_roll":       matched_roll,
                "match_confidence":   match_conf,
                "thumbnail":          ef["thumbnail"],
                "grid_cell":          ef.get("gridCell", ""),
                "seat_id":            ef["id"],
            })
        return faces

    def register_face(self, student_id, encoding_bytes):
        """Store raw bytes — used for backward compat."""
        pass

    def load_registered_faces(self, student_list):
        """Load all registered students with face encodings into InsightFace engine."""
        for s in student_list:
            if s.face_encoding and len(s.face_encoding) >= 512 * 4:
                face_rec_engine.load_from_bytes(
                    s.id, s.name, s.roll_no, s.face_encoding
                )
        print(f"FaceEngine: loaded {len(face_rec_engine.registered)} face embeddings.")


face_engine = FaceEngine()
