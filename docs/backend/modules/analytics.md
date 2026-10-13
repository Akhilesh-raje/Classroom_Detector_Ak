# 📊 Analytics & LLM Subsystem (`backend/core/analytics/`)

The Analytics subsystem generates longitudinal session reports and leverages local multimodal Large Language Models (LLMs) via **Ollama** to synthesize human-readable pedagogical insights.

---

## 📂 Subsystem File Map

| File | Primary Class | Responsibility |
| :--- | :--- | :--- |
| [report_generator.py](file:///c:/Users/rajea/Documents/classroom/backend/core/analytics/report_generator.py) | `ReportGenerator` | Compiles per-student analytics dictionaries and persists `SessionReport` records. |
| [ollama_labeler.py](file:///c:/Users/rajea/Documents/classroom/backend/core/analytics/ollama_labeler.py) | `OllamaLabeler` | Interfaces with local Ollama daemon (`gemma:2b`, `llama3`, `mistral`, `llava`). |

---

## 📈 Report Generation Flow (`ReportGenerator`)

At the conclusion of a session or video processing run, `generate_student_report()` builds a serializable JSON payload for every seated student:

```json
{
  "student_name": "Aarav Sharma",
  "roll_no": "CS2026-001",
  "grid_label": "R1C1",
  "session_id": "c7a8b9e1-23d4-4f5a-8b1e-6c9f2a4b8d0e",
  "session_date": "2026-10-13",
  "session_duration_seconds": 466.2,
  "avg_attention": 84.5,
  "peak_attention": 93.2,
  "low_attention": 67.4,
  "phone_usage_seconds": 0.0,
  "laptop_usage_seconds": 124.5,
  "earphone_usage_seconds": 0.0,
  "timeline": [
    {"activity": "studying", "start_ts": 0.0, "end_ts": 312.4, "duration": 312.4},
    {"activity": "talking", "start_ts": 312.4, "end_ts": 334.1, "duration": 21.7},
    {"activity": "attentive", "start_ts": 334.1, "end_ts": 466.2, "duration": 132.1}
  ]
}
```

---

## 🤖 Ollama Local LLM Integration (`OllamaLabeler`)

The system integrates with a local Ollama server running at `http://localhost:11434`:

1. **Model Discovery**: Queries `/api/tags` to auto-discover installed models (`gemma:2b`, `llama3`, `mistral`, `phi`).
2. **Graceful Fallback**: If Ollama is offline or not installed, the system falls back to rule-based algorithmic summaries without interrupting execution.
3. **Classroom Diagnostics Prompting**:
   - Compiles class attention distributions, distraction hot spots, and device usage statistics into a structured prompt.
   - Generates actionable advice for instructors (e.g. identifying disengaged rows or suggesting teaching pauses).
