import urllib.request
import urllib.parse
import json
import time

API = "http://127.0.0.1:8000/api/attendance"

# Observations simulating 11 students:
# 8 with verified biometric face matches (Tier 1)
# 3 seat-based candidates (Tier 2, no face match)
# 1 occupant in unassigned position (Tier 3)
obs = [
    {"track_id": 1, "seat_id": "R1C1", "matched_student_id": 1, "match_confidence": 0.92},
    {"track_id": 2, "seat_id": "R1C2", "matched_student_id": 2, "match_confidence": 0.88},
    {"track_id": 3, "seat_id": "R1C3", "matched_student_id": 3, "match_confidence": 0.85},
    {"track_id": 4, "seat_id": "R1C4", "matched_student_id": 4, "match_confidence": 0.94},
    {"track_id": 5, "seat_id": "R1C5", "matched_student_id": 5, "match_confidence": 0.89},
    {"track_id": 6, "seat_id": "R1C6", "matched_student_id": 6, "match_confidence": 0.86},
    {"track_id": 7, "seat_id": "R2C1", "matched_student_id": 7, "match_confidence": 0.90},
    {"track_id": 8, "seat_id": "R2C2", "matched_student_id": 8, "match_confidence": 0.91},
    {"track_id": 9, "seat_id": "R2C4", "matched_student_id": None, "match_confidence": 0.0},
    {"track_id": 10, "seat_id": "R2C5", "matched_student_id": None, "match_confidence": 0.0},
    {"track_id": 11, "seat_id": "R2C6", "matched_student_id": None, "match_confidence": 0.0},
]

print("Simulating 6 frames of continuous presence to pass temporal confirmation threshold...")
for frame_idx in range(1, 7):
    data = urllib.parse.urlencode({
        "observations": json.dumps(obs),
        "dry_run": "false"
    }).encode("utf-8")
    req = urllib.request.Request(f"{API}/process-observation", data=data)
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"Frame {frame_idx}: processed {res.get('processed')} observations.")
    time.sleep(0.05)

print("\nFetching Attendance Summary...")
req = urllib.request.Request(API)
with urllib.request.urlopen(req) as resp:
    summary = json.loads(resp.read().decode())
    print("\nAttendance Summary Counts:")
    print(json.dumps(summary["counts"], indent=2))
    print(f"\nReview Queue length: {len(summary.get('reviewQueue', []))}")
    for item in summary.get('reviewQueue', []):
        print(f" - Seat: {item.get('seatId')}, Candidate: {item.get('candidateName')}, State: {item.get('presenceState')}")
    print(f"\nRecent Events length: {len(summary.get('recentEvents', []))}")
