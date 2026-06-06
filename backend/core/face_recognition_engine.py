"""
SmartClass AI — Real Face Recognition Engine
Tries InsightFace → face_recognition → histogram fallback.
Never crashes regardless of which libraries are installed.
"""
from __future__ import annotations
import numpy as np
import cv2
import logging

log = logging.getLogger("face_rec")

# ── Try to load best available backend ───────────────────────────────────────
_backend = "histogram"  # default fallback

try:
    import insightface
    from insightface.app import FaceAnalysis
    _insight_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    _insight_app.prepare(ctx_id=0, det_size=(160, 160))
    _backend = "insightface"
    log.info("Face recognition backend: InsightFace (buffalo_sc)")
except Exception:
    try:
        import face_recognition as _fr_lib
        _backend = "face_recognition"
        log.info("Face recognition backend: face_recognition library")
    except Exception:
        log.info("Face recognition backend: histogram fallback (install insightface or face_recognition for better accuracy)")


class FaceRecognitionEngine:
    """
    Unified face recognition engine with automatic backend selection.
    InsightFace > face_recognition > histogram fallback.
    """

    @property
    def backend(self) -> str:
        """Return the name of the active face recognition backend."""
        return _backend

    @property
    def is_reliable(self) -> bool:
        """Histogram fallback is NOT reliable for identity matching."""
        return _backend != "histogram"

    def extract_encoding(self, face_crop_bgr) -> np.ndarray | None:
        """Extract face embedding from a BGR crop. Returns None if no face found."""
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None

        try:
            if _backend == "insightface":
                return self._encode_insightface(face_crop_bgr)
            elif _backend == "face_recognition":
                return self._encode_face_recognition(face_crop_bgr)
            else:
                return self._encode_histogram(face_crop_bgr)
        except Exception as e:
            log.debug(f"Encoding failed: {e}")
            return None

    def compare(self, enc1: np.ndarray, enc2: np.ndarray) -> float:
        """Cosine similarity between two encodings. Returns 0.0–1.0."""
        if enc1 is None or enc2 is None:
            return 0.0
        try:
            if _backend == "face_recognition":
                # face_recognition uses Euclidean distance — convert to similarity
                dist = float(np.linalg.norm(enc1 - enc2))
                return max(0.0, 1.0 - dist / 1.2)
            else:
                # Cosine similarity
                n1 = np.linalg.norm(enc1)
                n2 = np.linalg.norm(enc2)
                if n1 == 0 or n2 == 0:
                    return 0.0
                return float(np.dot(enc1, enc2) / (n1 * n2))
        except Exception:
            return 0.0

    def find_best_match(
        self,
        encoding: np.ndarray,
        registered: list[tuple[int, np.ndarray]],
        threshold: float = 0.75,
    ) -> tuple[int | None, float]:
        """
        Find best matching student from registered list.
        registered: list of (student_id, encoding)
        Returns (student_id, confidence) or (None, 0.0)
        """
        if encoding is None or not registered:
            return None, 0.0

        best_id   = None
        best_conf = 0.0

        for student_id, reg_enc in registered:
            if reg_enc is None:
                continue
            conf = self.compare(encoding, reg_enc)
            if conf > best_conf:
                best_conf = conf
                best_id   = student_id

        if best_conf >= threshold:
            return best_id, round(best_conf, 4)
        return None, round(best_conf, 4)

    def encoding_to_bytes(self, enc: np.ndarray) -> bytes:
        """Serialize encoding to bytes for DB storage."""
        return enc.astype(np.float32).tobytes()

    def bytes_to_encoding(self, data: bytes) -> np.ndarray | None:
        """Deserialize encoding from DB bytes."""
        if not data:
            return None
        try:
            arr = np.frombuffer(data, dtype=np.float32)
            return arr
        except Exception:
            return None

    # ── Backend implementations ───────────────────────────────────────────────

    def _encode_insightface(self, bgr) -> np.ndarray | None:
        faces = _insight_app.get(bgr)
        if not faces:
            return None
        # Use largest face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        return face.embedding.astype(np.float32)

    def _encode_face_recognition(self, bgr) -> np.ndarray | None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        encs = _fr_lib.face_encodings(rgb)
        if not encs:
            return None
        return np.array(encs[0], dtype=np.float32)

    def _encode_histogram(self, bgr) -> np.ndarray | None:
        """Simple color histogram as fallback encoding (64-bin per channel)."""
        resized = cv2.resize(bgr, (64, 64))
        hist = []
        for ch in range(3):
            h = cv2.calcHist([resized], [ch], None, [64], [0, 256])
            cv2.normalize(h, h)
            hist.extend(h.flatten().tolist())
        return np.array(hist, dtype=np.float32)


# Singleton
face_rec_engine = FaceRecognitionEngine()
