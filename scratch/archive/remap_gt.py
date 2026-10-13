import json
import os

gt_path = os.path.join("backend", "tests", "ground_truth_classroom.json")
with open(gt_path, "r", encoding="utf-8") as f:
    gt = json.load(f)

REMAP = {
    "R2C1": "R1C1",  # trk6 (front-left)
    "R1C1": "R1C2",  # trk5
    "R1C2": "R1C3",  # trk8
    "R1C3": "R1C4",  # trk4
    "R1C4": "R1C5",  # trk3
    "R1C5": "R1C6",  # trk2 (front-right)
    "R2C2": "R2C1",  # trk10 (back-left)
    "R2C3": "R2C2",  # trk9
    "R2C4": "R2C3",  # trk12
    "R2C5": "R2C4",  # trk7
    "R2C6": "R2C5",  # trk11
    "R1C6": "R2C6",  # trk1 (back-right)
}

for frame_idx, annos in gt["frames"].items():
    print(f"Frame {frame_idx}:")
    for a in annos:
        old_s = a.get("assigned_seat", "")
        new_s = REMAP.get(old_s, old_s)
        a["assigned_seat"] = new_s
        print(f"  {old_s:10s} -> {new_s:10s} bbox: {a['bbox']}")

# Save updated ground truth
with open(gt_path, "w", encoding="utf-8") as f:
    json.dump(gt, f, indent=2)

print("\nSuccessfully updated backend/tests/ground_truth_classroom.json!")
