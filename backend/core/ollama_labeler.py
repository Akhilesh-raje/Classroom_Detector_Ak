"""
SmartClass AI — Ollama Local LLM Seat Labeler
Uses Ollama (Gemma/Llama/Mistral) to intelligently label seats and
generate classroom analysis. Falls back gracefully if Ollama is offline.
"""
from __future__ import annotations
import requests
import json
import logging

log = logging.getLogger("ollama_labeler")

OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODELS = ["gemma:2b", "gemma", "llama2", "mistral", "phi"]


class OllamaLabeler:

    def __init__(self, model: str | None = None):
        self.model   = model
        self._ready  = None   # None = not checked yet

    def _get_model(self) -> str | None:
        """Find first available Ollama model."""
        if self.model:
            return self.model
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                available = [m["name"] for m in r.json().get("models", [])]
                for m in OLLAMA_MODELS:
                    for a in available:
                        if m.split(":")[0] in a:
                            self.model = a
                            return a
        except Exception:
            pass
        return None

    def _call_ollama(self, prompt: str) -> str | None:
        """Call Ollama API. Returns response text or None if unavailable."""
        model = self._get_model()
        if not model:
            return None
        try:
            r = requests.post(
                OLLAMA_URL,
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=30,
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip()
        except Exception as e:
            log.debug(f"Ollama call failed: {e}")
        return None

    def label_seats(
        self,
        grid_cells: list[str],
        student_names: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Map grid cell labels to student display names.
        If student_names provided, assigns them to cells in order.
        Uses Ollama to generate smart labels if no names given.
        Returns dict: {"R1C1": "Stud1", "R1C2": "Stud2", ...}
        """
        if student_names and len(student_names) >= len(grid_cells):
            # Direct assignment — names provided
            return {cell: student_names[i] for i, cell in enumerate(grid_cells)}

        # Try Ollama for smart labeling
        if student_names:
            prompt = (
                f"You are labeling classroom seats. "
                f"Grid cells: {grid_cells}. "
                f"Students: {student_names}. "
                f"Assign each student to the most logical seat based on typical classroom seating. "
                f"Return ONLY a JSON object like {{\"R1C1\": \"Alice\", \"R1C2\": \"Bob\"}}. "
                f"No explanation, just JSON."
            )
            response = self._call_ollama(prompt)
            if response:
                try:
                    # Extract JSON from response
                    start = response.find("{")
                    end   = response.rfind("}") + 1
                    if start >= 0 and end > start:
                        mapping = json.loads(response[start:end])
                        # Validate all cells are present
                        if all(c in mapping for c in grid_cells):
                            return mapping
                except Exception:
                    pass

        # Fallback: auto-generate Stud1, Stud2...
        return self._fallback_label(grid_cells, student_names)

    def analyze_classroom(self, student_reports: list[dict]) -> str:
        """
        Generate a natural language classroom analysis from student reports.
        Returns a summary string.
        """
        if not student_reports:
            return "No student data available for analysis."

        # Build summary stats for the prompt
        avg_attns = [r.get("avg_attention", 0) for r in student_reports]
        class_avg = sum(avg_attns) / max(1, len(avg_attns))
        studying  = sum(1 for r in student_reports
                        if r.get("activity_timeline") and
                        r["activity_timeline"][-1].get("activity") in ("studying", "attentive"))
        distracted = sum(1 for r in student_reports
                         if r.get("activity_timeline") and
                         r["activity_timeline"][-1].get("activity") in ("distracted", "phone", "drowsy"))
        phone_users = [r.get("grid_label", r.get("student_name", "?"))
                       for r in student_reports
                       if r.get("electronics", {}).get("phone_duration_seconds", 0) > 5]

        per_student = ", ".join(
            f"{r.get('grid_label','?')}:{r.get('avg_attention',0):.0f}%"
            for r in student_reports
        )
        prompt = (
            f"You are a classroom AI assistant. Analyze this classroom session:\n"
            f"- Total students: {len(student_reports)}\n"
            f"- Class average attention: {class_avg:.1f}%\n"
            f"- Students studying/attentive: {studying}\n"
            f"- Students distracted: {distracted}\n"
            f"- Phone usage detected: {phone_users if phone_users else 'None'}\n"
            f"- Per-student attention: {per_student}\n\n"
            f"Write a brief 3-4 sentence classroom report for the teacher. "
            f"Be specific about which seats need attention. Keep it professional and concise."
        )

        response = self._call_ollama(prompt)
        if response:
            return response

        # Fallback summary
        return self._fallback_analysis(student_reports, class_avg, studying, distracted, phone_users)

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        return self._get_model() is not None

    def get_model(self) -> str | None:
        """Public accessor for current model name."""
        return self._get_model()

    def _fallback_label(self, grid_cells, student_names=None) -> dict:
        result = {}
        for i, cell in enumerate(grid_cells):
            if student_names and i < len(student_names):
                result[cell] = student_names[i]
            else:
                result[cell] = f"Stud{i + 1}"
        return result

    def _fallback_analysis(self, reports, class_avg, studying, distracted, phone_users) -> str:
        n = len(reports)
        lines = [
            f"Class session summary: {n} students monitored, average attention {class_avg:.1f}%.",
            f"{studying} student(s) were actively studying, {distracted} showed signs of distraction.",
        ]
        if phone_users:
            lines.append(f"Phone usage detected at seats: {', '.join(str(p) for p in phone_users)}.")
        low = [r for r in reports if r.get("avg_attention", 100) < 60]
        if low:
            seats = [r.get("grid_label", "?") for r in low]
            lines.append(f"Seats needing attention: {', '.join(seats)}.")
        return " ".join(lines)


# Singleton
ollama_labeler = OllamaLabeler()
