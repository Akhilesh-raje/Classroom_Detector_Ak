import csv

with open("scratch/benchmark_full_extent_300f.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    f_num = int(r["frame"])
    if f_num > 10:
        break
    if r["track_id"] in ("5", "6", "4"):
        print(f"f={f_num:2d} tid={r['track_id']} head=({float(r['head_x']):5.1f},{float(r['head_y']):5.1f}) seat={r['seat']:6s} cost={float(r['cost']):5.1f} prev={r.get('prev_seat')}")
