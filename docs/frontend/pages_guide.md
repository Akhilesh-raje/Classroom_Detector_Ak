# 🖥️ Frontend Pages Guide

This document details the functionality, components, user workflows, and state interactions across all 9 pages of the SmartClass AI dashboard.

---

## 📑 Page Breakdown

### 1. All-In-One Hub (`/dashboard` — [ClassroomApp.jsx](file:///c:/Users/rajea/Documents/classroom/src/pages/ClassroomApp.jsx))
- **Purpose**: The central command deck for live classroom monitoring during an active lecture.
- **Key Features**:
  - Split viewport: Live MJPEG camera feed with computer vision HUD on the left; real-time seating matrix on the right.
  - Live metric chips: Active Headcount, Class Average Attentiveness, Focus Index, Distraction Count.
  - Quick action toolbar: Camera toggle, Session Start/Stop, and full-screen presentation mode.

### 2. Live Attendance (`/attendance` — [Attendance.jsx](file:///c:/Users/rajea/Documents/classroom/src/pages/Attendance.jsx))
- **Purpose**: Real-time continuous attendance monitoring and manual review interface.
- **Key Features**:
  - Live Verification Grid: Shows enrolled students with badges (`VERIFIED`, `CANDIDATE`, `FLAGGED`, `ABSENT`).
  - Candidate Review Queue: Highlights borderline matches (Tier 2) or unknown occupants (Tier 3) with captured face crops.
  - Action buttons: One-click Confirm, Reject, or Manual Mark.
  - Export: Direct CSV attendance ledger download button.

### 3. Classroom Seating (`/classroom` — [Classroom.jsx](file:///c:/Users/rajea/Documents/classroom/src/pages/Classroom.jsx))
- **Purpose**: Physical floor plan layout and seat occupancy heatmaps.
- **Key Features**:
  - Interactive Desk Grid: Visual representation of classroom benches (e.g., Row 1 Benches 1–3, Left/Right positions).
  - Desk Occupancy States: Green (Occupied & Attentive), Yellow (Talking/Distracted), Grey (Empty).
  - Drag-and-drop or modal student-to-seat assignment.

### 4. Structure AI Lab (`/structure-lab` — [StructureLab.jsx](file:///c:/Users/rajea/Documents/classroom/src/pages/StructureLab.jsx))
- **Purpose**: Laboratory environment for inspecting and configuring 3D geometric scene understanding.
- **Key Features**:
  - Structural Perception Gallery: Displays the 8 pipeline stages (`00_clean_frame.jpg` through `06_validation_overlay.jpg`).
  - Monocular Depth Visualizer: Interactive depth map inspector showing 3D plane estimations.
  - Topology Calibration: Trigger on-demand Hough line desk edge discovery.

### 5. Video Engine (`/video-engine` — [TestVideo.jsx](file:///c:/Users/rajea/Documents/classroom/src/pages/TestVideo.jsx))
- **Purpose**: Offline video analysis and multi-modal expression testing playground.
- **Key Features**:
  - Local Video Selector: Choose between test videos (`real classroom.mp4`, `student sitting.mp4`).
  - Playback Controls: Frame-by-frame stepping, scrubber, and playback speed multiplier.
  - Expression & Pose Inspector: Detailed facial landmark inspector and blendshape radar.

### 6. Student Management (`/students` — [Students.jsx](file:///c:/Users/rajea/Documents/classroom/src/pages/Students.jsx))
- **Purpose**: Institutional student roster and facial enrollment hub.
- **Key Features**:
  - Student Directory: Searchable, paginated table of registered students with roll numbers and sections.
  - Face Enrollment Modal: Single-photo upload or multi-shot camera capture with real-time embedding extraction feedback.
  - Roster Import/Export: Add new students or remove departed students.

### 7. Student Profile (`/students/:id` — [StudentProfile.jsx](file:///c:/Users/rajea/Documents/classroom/src/pages/StudentProfile.jsx))
- **Purpose**: In-depth longitudinal analytics for an individual student.
- **Key Features**:
  - Longitudinal Attention Curve: Historical session attentiveness charts over time.
  - Behavioral Breakdown: Pie chart of activities (`studying`, `attentive`, `talking`, `distracted`).
  - Attendance Consistency: Total attended sessions vs total lectures held.

### 8. Sessions Audit (`/sessions` — [Sessions.jsx](file:///c:/Users/rajea/Documents/classroom/src/pages/Sessions.jsx))
- **Purpose**: Historical lecture session audit log and timeline player.
- **Key Features**:
  - Session Archive: List of past recorded lectures with dates, durations, and average attendance.
  - Detailed Audit View: Drill into any session to see per-student timelines and distraction spikes.

### 9. Analytics & Reports (`/reports` — [Reports.jsx](file:///c:/Users/rajea/Documents/classroom/src/pages/Reports.jsx))
- **Purpose**: Institutional reporting, PDF generation, and LLM pedagogical insight delivery.
- **Key Features**:
  - Class-Wide Trends: Longitudinal attentiveness graphs across semesters or subjects.
  - Ollama LLM Recommendations: Generates AI instructional suggestions for the instructor.
  - PDF/CSV Export: Download official academic reports.
