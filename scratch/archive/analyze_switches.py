import csv
from collections import defaultdict

with open("benchmark_association_log.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total rows in log: {len(rows)}")

# 1. Inspect UNASSIGNED tracks
unassigned_by_track = defaultdict(list)
for r in rows:
    if r["status"] == "UNASSIGNED":
        unassigned_by_track[r["track_id"]].append(r)

print(f"\nUNASSIGNED count: {sum(len(v) for v in unassigned_by_track.values())}")
for tid, obs in unassigned_by_track.items():
    print(f"Track {tid}: {len(obs)} unassigned frames. Sample: head_x={obs[0]['head_x']}, head_y={obs[0]['head_y']}, cost={obs[0]['cost']}, row={obs[0]['row']}")

# 2. Inspect Seat Occupants & Switches
seat_occupants = defaultdict(list)
for r in rows:
    if r["seat"] != "UNASSIGNED" and r["track_id"] != "GHOST":
        seat_occupants[r["seat"]].append((int(r["frame"]), r["track_id"]))

print("\nSeat occupancy sequences:")
for seat, occ in sorted(seat_occupants.items()):
    # Find transitions
    transitions = []
    prev_tid = None
    for f_idx, tid in occ:
        if tid != prev_tid:
            transitions.append((f_idx, tid))
            prev_tid = tid
    print(f"Seat {seat}: {len(transitions)-1} switches -> {transitions}")
