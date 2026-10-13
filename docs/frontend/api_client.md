# 🔌 Frontend API Client (`src/api.js`)

The frontend communicates with the FastAPI backend using a clean, Promise-based API client defined in [src/api.js](file:///c:/Users/rajea/Documents/classroom/src/api.js).

---

## 🏛️ Client Structure

```javascript
const API = "http://localhost:8000/api";

export const api = {
  // Domain method groups:
  // - Students & Profiles
  // - Biometrics & Enrollment
  // - Attendance Decision Engine
  // - Camera Hardware & Streaming
  // - Sessions & Historical Reports
  // - Structure AI Laboratory
  // - Ollama Multimodal LLM
};
```

---

## 📚 Method Catalog

### 👥 Students
- `api.getStudents()`: Fetches all registered student records.
- `api.createStudent(formData)`: Submits new student with profile data and photo.
- `api.deleteStudent(id)`: Removes student record and biometric embeddings.
- `api.assignSeat(studentId, gridCell)`: Updates student primary seat binding.
- `api.getStudentProfile(studentId)`: Fetches historical profile analytics.

### 👤 Biometrics & Face Enrollment
- `api.registerFace(studentId, imageBlob)`: Uploads single face portrait.
- `api.enrollFace(studentId, formData)`: Multi-shot face enrollment with multiple angles.
- `api.processFrame(imageBlob)`: Sends camera snapshot for ad-hoc face landmark verification.

### 📋 Attendance Engine
- `api.getAttendance(date)`: Queries active attendance records.
- `api.processObservations(observations, dryRun)`: Pushes frame observations to temporal confirmation engine.
- `api.reviewAttendance(recordId, decision, assignedStudentId)`: Submits instructor review for Tier-2/3 flags.
- `api.markManual(studentId, status)`: Overrides attendance manually.
- `api.resetAttendance(targetDate, dryRunOnly)`: Clears active attendance buffer.
- `api.exportAttendanceUrl(date)`: Returns direct URL for CSV download.

### 📹 Camera & Telemetry
- `api.startCamera(deviceIndex)`: Powers on physical camera.
- `api.stopCamera()`: Shuts down camera capture.
- `api.getDetections()`: Retrieves latest bounding boxes and track states.
- `api.getLiveTelemetry()`: Retrieves real-time attention scores and activity counts.

### 🏗️ Structure AI & Sessions
- `api.getStructureArtifacts()`: Fetches geometric stage images and topology JSON.
- `api.runStructurePerception()`: Triggers keyframe depth and bench edge analysis.
- `api.startSession(sessionName)`: Initiates tracked lecture session.
- `api.stopSession()`: Finalizes session and builds `SessionReport`.
- `api.getSessions()`: Queries list of past lectures.
- `api.getSessionReport(sessionId)`: Retrieves complete analytics JSON.

### 🤖 Ollama LLM
- `api.checkOllamaStatus()`: Tests connection to local Ollama daemon.
- `api.getOllamaAnalysis()`: Requests AI instructional advice from Ollama.
