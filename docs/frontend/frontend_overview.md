# 💻 Frontend Architecture Overview

SmartClass AI features a high-density, command-center dashboard built with **React 18**, **Vite**, and **React Router 6**. It uses a dark glassmorphic design system tailored for high-contrast classroom monitoring, live video overlays, and telemetry dashboards.

---

## 🎨 Design System & Aesthetics

Implemented in [src/index.css](file:///c:/Users/rajea/Documents/classroom/src/index.css), the UI utilizes custom CSS variables, translucent glass cards, and neon telemetry status accents:

- **Surface Colors**: Deep charcoal and cosmic navy (`#04101e`, `#0a192f`, `#112240`).
- **Glassmorphism**: Backdrop blur filters (`backdrop-filter: blur(12px)`), subtle 1px border glows (`rgba(255, 255, 255, 0.08)`).
- **Accents**:
  - Cyan / Electric Blue (`#38bdf8`): Active navigation and primary interactions.
  - Emerald Green (`#22c55e`): High attention, verified attendance, and online cameras.
  - Amber Yellow (`#f59e0b`): Candidate reviews, talking, and moderate attention.
  - Crimson Red (`#ef4444`): Drowsiness alerts, phone distractions, and unverified visitors.

---

## 🧭 Navigation & Layout

Managed in [src/App.jsx](file:///c:/Users/rajea/Documents/classroom/src/App.jsx):

```mermaid
graph TD
    APP["React Root (App.jsx)"] --> LAYOUT["Master Layout"]
    LAYOUT --> SIDEBAR["Sidebar Navigation (Command Center)"]
    LAYOUT --> TOPBAR["Topbar (Live Clock, Session Status, Title)"]
    LAYOUT --> ROUTES["Route Views"]

    ROUTES --> P1["/dashboard : All-In-One Hub (ClassroomApp.jsx)"]
    ROUTES --> P2["/attendance : Live Attendance (Attendance.jsx)"]
    ROUTES --> P3["/classroom : Seating Grid & Radar (Classroom.jsx)"]
    ROUTES --> P4["/structure-lab : Structure AI Lab (StructureLab.jsx)"]
    ROUTES --> P5["/video-engine : Video Analysis Engine (TestVideo.jsx)"]
    ROUTES --> P6["/students : Student Roster (Students.jsx)"]
    ROUTES --> P7["/students/:id : Profile Deep-Dive (StudentProfile.jsx)"]
    ROUTES --> P8["/sessions : Session Audit & History (Sessions.jsx)"]
    ROUTES --> P9["/reports : Longitudinal Analytics (Reports.jsx)"]
```

---

## 🔄 Real-Time State & Telemetry Ingestion

1. **Video Streaming**: Renders native `<img src="http://localhost:8000/api/camera/stream" />` or `/api/live-classroom/stream` with automatic retry on disconnect.
2. **Telemetry Heartbeat**: Uses `setInterval` polling (typically $250\text{ms} - 1000\text{ms}$) against `/api/live-classroom/telemetry` to update widget state without full-page re-renders.
3. **Decoupled Control**: Instructors can trigger camera controls, start sessions, review attendance, and assign seats with instant visual feedback.
