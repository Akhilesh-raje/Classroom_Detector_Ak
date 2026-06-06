const API = "http://localhost:8000/api";

export const api = {
  // ── Students ──────────────────────────────────────────────
  getStudents: () => fetch(`${API}/students`).then(r => r.json()),

  createStudent: (formData) =>
    fetch(`${API}/students`, { method: "POST", body: formData }).then(r => r.json()),

  deleteStudent: (id) =>
    fetch(`${API}/students/${id}`, { method: "DELETE" }).then(r => r.json()),

  assignSeat: (studentId, gridCell) => {
    const fd = new FormData();
    fd.append("grid_cell", gridCell);
    return fetch(`${API}/students/${studentId}/assign-seat`, { method: "POST", body: fd }).then(r => r.json());
  },

  // ── Face ──────────────────────────────────────────────────
  registerFace: (studentId, imageBlob) => {
    const fd = new FormData();
    fd.append("image", imageBlob, "face.jpg");
    return fetch(`${API}/register-face/${studentId}`, { method: "POST", body: fd }).then(r => r.json());
  },

  processFrame: (imageBlob) => {
    const fd = new FormData();
    fd.append("image", imageBlob, "frame.jpg");
    return fetch(`${API}/process-frame`, { method: "POST", body: fd }).then(r => r.json());
  },

  // ── Attendance ────────────────────────────────────────────
  getAttendance: (date) =>
    fetch(`${API}/attendance${date ? `?target_date=${date}` : ""}`).then(r => r.json()),

  markManual: (studentId, status) => {
    const fd = new FormData();
    fd.append("student_id", studentId);
    fd.append("status", status);
    return fetch(`${API}/attendance/manual`, { method: "POST", body: fd }).then(r => r.json());
  },

  // ── Analytics ─────────────────────────────────────────────
  getSummary: () => fetch(`${API}/analytics/summary`).then(r => r.json()),
  getActivityLogs: (limit = 50) => fetch(`${API}/activity-logs?limit=${limit}`).then(r => r.json()),

  // ── Sessions ──────────────────────────────────────────────
  getSessions: () => fetch(`${API}/sessions`).then(r => r.json()),

  startSession: (name = "") => {
    const fd = new FormData();
    fd.append("session_name", name);
    return fetch(`${API}/sessions/start`, { method: "POST", body: fd }).then(r => r.json());
  },

  stopSession: () => fetch(`${API}/sessions/stop`, { method: "POST" }).then(r => r.json()),

  // ── Reports ───────────────────────────────────────────────
  getSessionReports: (sessionId) =>
    fetch(`${API}/reports/session/${sessionId}`).then(r => r.json()),

  getStudentReports: (studentId) =>
    fetch(`${API}/reports/student/${studentId}`).then(r => r.json()),

  // ── Video Analysis ────────────────────────────────────────
  getVideoList: () => fetch(`${API}/video-list`).then(r => r.json()),

  analyzeVideo: (videoFile, frameSkip = 3) => {
    const fd = new FormData();
    fd.append("video", videoFile);
    fd.append("frameSkip", frameSkip);
    return fetch(`${API}/analyze-video`, { method: "POST", body: fd });
  },

  analyzeVideoPath: (path, frameSkip = 3) => {
    const fd = new FormData();
    fd.append("video_path", path);
    fd.append("frameSkip", frameSkip);
    return fetch(`${API}/analyze-video-path`, { method: "POST", body: fd });
  },

  // ── Ollama / Local AI ─────────────────────────────────────
  ollamaStatus: () => fetch(`${API}/ollama/status`, { method: "POST" }).then(r => r.json()),

  ollamaLabelSeats: () =>
    fetch(`${API}/ollama/label-seats`, { method: "POST" }).then(r => r.json()),

  ollamaAnalyze: () => fetch(`${API}/ollama/analyze`).then(r => r.json()),

  ollamaAnalyzeStudent: (seatId) =>
    fetch(`${API}/ollama/analyze-student/${seatId}`, { method: "POST" }).then(r => r.json()),

  // ── Camera ────────────────────────────────────────────────
  streamUrl: `${API}/camera/stream`,
  startCamera: () => fetch(`${API}/camera/start`, { method: "POST" }).then(r => r.json()),
  stopCamera:  () => fetch(`${API}/camera/stop`,  { method: "POST" }).then(r => r.json()),
  getDetections: () => fetch(`${API}/camera/detections`).then(r => r.json()),
};
