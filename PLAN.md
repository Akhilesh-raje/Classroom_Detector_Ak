# SmartClass AI — System Restructuring & Web App Overhaul Plan

## Objectives
1. **Directory Restructuring & Cleanup**:
   - Clean up repository root (move video files, benchmark scripts, loose test images, stray markdown reports, and YOLO weight files into structured directories).
   - Consolidate webcam runners and test scripts into a unified CLI (`cli.py`).
2. **Backend Architecture Upgrade**:
   - Expose the `--structure-ai` pipeline and `ComprehensiveExporter` via FastAPI endpoints.
   - Serve pipeline image artifacts (`outputs/structure_experiment/`) via static file mounts.
   - Stabilize live camera streaming and avoid OpenCV resource lock contention.
3. **Frontend Complete Overhaul (Modern Luxury Aesthetics & Functionality)**:
   - Establish cohesive design system matching `classroom_dashboard.html` (Google Font Outfit + JetBrains Mono, glassmorphism, glowing accents, dark space slate `#090d16`).
   - Add **Structure AI Lab** view displaying all 8 stage images (Clean Frame, Furniture Boxes, People, Furniture, Depth, Seats, Final Structure, Validation Overlay), Interactive Topology Inspector, and Student Dossier cards with face crops.
   - Rebuild **Live Classroom Grid** (`Classroom.jsx`) with live desk telemetry, real-time attention gauges, and video HUD.
   - Rebuild **Video Analysis Engine** (`TestVideo.jsx`) with synced overlays and pipeline execution.
   - Polish **Students**, **Sessions**, and **Reports** with consistent styling and export capabilities.

---

## Detailed Task Breakdown

### Phase 1: Directory Restructuring & Cleanup
- [x] Create structured directories:
  - `media/videos/` (for `.mp4` video recordings)
  - `models/` (ensure `yolo11s-seg.pt` is housed properly)
  - `benchmarks/` (consolidate all `benchmark_*.py`, `benchmark_*.csv`, and logs)
  - `docs/reports/` (move loose root `.md` reports except README.md)
  - `outputs/archive/` (archive root test outputs `v3_*.jpg`, `test_result.jpg`, etc.)
- [x] Unify CLI runner:
  - Create `cli.py` supporting `webcam`, `test --structure-ai`, `analyze-video`, `serve-backend`, `serve-frontend`.
- [x] Consolidate duplicate webcam scripts and organize scripts directory.

### Phase 2: Backend API Upgrades (`backend/main.py`)
- [x] Mount `outputs/` directory statically at `/outputs` for live frontend asset consumption.
- [x] Mount `media/videos/` directory statically at `/media-videos`.
- [x] Implement `GET /api/structure-ai/latest` to return:
  - Pipeline stage image URLs
  - Classroom topology nodes & desk layout
  - Session metrics & KPIs
  - Student dossiers with face crop URLs and attention metrics
- [x] Implement `POST /api/structure-ai/run` to invoke the `ComprehensiveExporter` pipeline.

### Phase 3: Frontend Aesthetics & Design System Overhaul
- [x] Update `index.html` with Google Fonts (`Outfit` + `JetBrains Mono`).
- [x] Rewrite `src/index.css` with a comprehensive dark mode glassmorphism token system.
- [x] Update `src/App.jsx` navigation bar to include:
  - **Structure AI Lab** (Stage Gallery, Topology Visualizer, Student Dossiers)
  - **Classroom Live** (Real-time Seating Grid & Webcam HUD)
  - **Video Engine** (Video Analysis & Overlay Inspector)
  - **Students & Roster**
  - **Sessions & Reports**

### Phase 4: Frontend Pages Implementation
- [x] Create `src/pages/StructureLab.jsx`:
  - 8-stage interactive pipeline image gallery with lightbox zoom
  - Interactive classroom topology grid & bench layout
  - Student Dossier grid showing face crops, attention scores, activity tags, phone/drowsy warnings
  - Quick action to trigger structure analysis on test video
- [x] Overhaul `src/pages/Classroom.jsx`:
  - Modern glass cards with Front (Cyan) / Back (Purple) row telemetry
  - Accurate desk & bench grid representation
- [x] Rebuild `src/pages/TestVideo.jsx`:
  - Replaced broken Tailwind styles with native responsive layout
  - Added dual mode: Deep Structure AI Pipeline vs Real-Time Scan
  - Synced canvas overlays

### Phase 5: Verification & End-to-End Testing
- [x] Verify frontend build (`npm run build` completed in 4.88s).
- [x] Test backend endpoints (`/api/structure-ai/latest`, `/api/video-list`, `/outputs/` verified 200 OK).
- [x] Validate CLI commands (`python cli.py --help` verified).

### Phase 6: Full Integration Enhancements (Completed)
- [x] **Legacy Scratch Cleanup**: All 52 loose files in `scratch/` moved to `scratch/archive/`.
- [x] **Webcam Structure AI Overlay**: Overlaid locked spatial seat anchors (`R1C1..R2C6`) onto the live MJPEG camera feed and associated detected occupants to physical seat IDs.
- [x] **Physical Seat Prioritization in Enrollment**: Configured student seat picker to prioritize validated Structure AI desk nodes.
- [x] **Reporting & Exporting**: Added direct **CSV Attendance Export** and **Print/PDF Report Generation** to the Reports view.
