# 🚀 Setup, Installation & Operations Guide

This guide provides end-to-end instructions for installing, configuring, and operating **SmartClass AI** on Windows, Linux, or macOS.

---

## 💻 Hardware & System Requirements

### Recommended Hardware (Real-time Live Classroom)
- **CPU**: Intel Core i5/i7 (11th Gen+) or AMD Ryzen 5/7 (3000 series+).
- **GPU**: NVIDIA RTX 3060 / 4060 (6GB+ VRAM) with CUDA 11.8 or 12.x.
- **RAM**: 16 GB DDR4/DDR5.
- **Camera**: USB 3.0 / RTSP 1080p or 4K wide-angle camera mounted at front/side corner of classroom.
- **Storage**: 10 GB free space for models and video caches.

### Minimum Hardware (Offline Video Analysis / Testing)
- **CPU**: Quad-core modern processor.
- **RAM**: 8 GB.
- **GPU**: Integrated Graphics (runs in CPU fallback mode).

---

## 📦 Prerequisites

1. **Python**: Version `3.10`, `3.11`, or `3.12`.
2. **Node.js**: Version `18.x` or `20.x` LTS + `npm`.
3. **Ollama** *(Optional, for AI insights)*: Download from [ollama.ai](https://ollama.ai).

---

## 🛠️ Step-by-Step Installation

### 1. Clone & Setup Python Environment
```bash
# Clone the repository
git clone https://github.com/Akhilesh-raje/classroom-detection.git
cd classroom-detection

# Create Python virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on Linux/macOS:
source .venv/bin/activate

# Install backend dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 2. Verify AI Model Weights
Ensure required model assets exist:
- `yolo11s.pt` (automatically downloaded by Ultralytics on first run).
- `models/face_landmarker.task` (pre-bundled in repository, ~3.6 MB).

### 3. Install Frontend Dependencies
```bash
# From workspace root
npm install
```

### 4. Setup Local Ollama LLM (Optional)
```bash
# Install Ollama, then pull recommended models:
ollama pull gemma:2b
# OR
ollama pull llama3
```

---

## 🚀 Running the System

You can run the full platform using the unified CLI ([cli.py](file:///c:/Users/rajea/Documents/classroom/cli.py)) or launch services independently.

### Option A: Using the Master CLI
```bash
# Terminal 1: Launch FastAPI Backend
python cli.py backend --port 8000

# Terminal 2: Launch React Frontend
python cli.py frontend
```

### Option B: Manual Service Launch
```bash
# Terminal 1: Backend
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
npm run dev
```

The React dashboard will be live at:
👉 **`http://localhost:5173`** (or `http://localhost:3000`)

The FastAPI Swagger interactive documentation will be live at:
👉 **`http://localhost:8000/docs`**

---

## 🔧 Production Building & Serving

To generate an optimized frontend production bundle:
```bash
npm run build
```
This builds static HTML, JS, and CSS into the `dist/` folder, which can be served directly by Nginx or mounted as static files in FastAPI.

---

## ❓ Troubleshooting & FAQs

### 1. Camera opens in low resolution ($640 \times 480$)
- In `run_webcam.py`, `open_camera_optimized()` forces `cv2.CAP_DSHOW` and explicitly requests $3840 \times 2160$ before falling back to $1920 \times 1080$. Ensure your USB port supports USB 3.0 throughput.

### 2. CUDA Out of Memory (OOM)
- Switch model to nano: in `backend/core/config.py`, change `YOLO_MODEL_PATH = "yolo11n.pt"`.
- Increase `FRAME_SKIP` in video processing to `4` or `5`.

### 3. Face matching returns "Unknown"
- Ensure students have enrolled face photos in the **Students** tab or via `register_test_students.py`.
- Check lighting in classroom to ensure faces are not back-lit or heavily shadowed.
