"""
SmartClass AI — Universal Seat Naming
======================================
Handles two classroom layouts:

  bench layout  →  "R1B2-L" = Row 1, Bench 2, Left seat
  desk layout   →  "R1C2"   = Row 1, Column 2  (backward-compatible)

Position codes for bench layout:
  1 seat/bench  →  ["C"]          (centre only)
  2 seats/bench →  ["L", "R"]
  3 seats/bench →  ["L", "M", "R"]
  4 seats/bench →  ["L", "ML", "MR", "R"]

The canonical seat label is always a single string that unambiguously
identifies a physical seat in any classroom configuration.
"""
from __future__ import annotations

_POSITIONS = {
    1: ["C"],
    2: ["L", "R"],
    3: ["L", "M", "R"],
    4: ["L", "ML", "MR", "R"],
}


def seat_label(row: int, bench: int, pos: str | None, layout_type: str = "bench") -> str:
    """
    Return the canonical seat label string.

    bench layout:  seat_label(1, 2, "L", "bench") → "R1B2-L"
    desk layout:   seat_label(1, 2, None, "desk")  → "R1C2"
    """
    if layout_type == "desk" or pos is None:
        return f"R{row}C{bench}"
    return f"R{row}B{bench}-{pos}"


def parse_seat_label(label: str) -> dict:
    """
    Parse a canonical seat label back to its components.
    Returns {"row", "bench", "pos", "layout_type"}.
    """
    import re
    # bench format: R1B2-L
    m = re.fullmatch(r"R(\d+)B(\d+)-([A-Z]+)", label)
    if m:
        return {"row": int(m.group(1)), "bench": int(m.group(2)),
                "pos": m.group(3), "layout_type": "bench"}
    # desk format: R1C2
    m = re.fullmatch(r"R(\d+)C(\d+)", label)
    if m:
        return {"row": int(m.group(1)), "bench": int(m.group(2)),
                "pos": None, "layout_type": "desk"}
    return {"row": None, "bench": None, "pos": None, "layout_type": "unknown"}


def all_seats(rows: int, benches_per_row: int, seats_per_bench: int,
              layout_type: str = "bench") -> list[str]:
    """
    Return the ordered list of all seat labels for a given classroom config.
    Order: row-major, left-to-right within each bench.
    """
    positions = _POSITIONS.get(seats_per_bench, ["C"])
    labels = []
    for r in range(1, rows + 1):
        for b in range(1, benches_per_row + 1):
            if layout_type == "desk":
                labels.append(seat_label(r, b, None, "desk"))
            else:
                for p in positions:
                    labels.append(seat_label(r, b, p, "bench"))
    return labels


def seats_per_row_labels(row: int, benches_per_row: int, seats_per_bench: int,
                         layout_type: str = "bench") -> list[str]:
    """All seat labels for a single row."""
    positions = _POSITIONS.get(seats_per_bench, ["C"])
    if layout_type == "desk":
        return [seat_label(row, b, None, "desk") for b in range(1, benches_per_row + 1)]
    return [seat_label(row, b, p, "bench")
            for b in range(1, benches_per_row + 1) for p in positions]


# ── Self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # bench 4 rows × 3 benches × 2 seats = 24 seats
    bench_seats = all_seats(4, 3, 2, "bench")
    assert len(bench_seats) == 24, f"expected 24, got {len(bench_seats)}"
    assert bench_seats[0] == "R1B1-L"
    assert bench_seats[1] == "R1B1-R"
    assert bench_seats[2] == "R1B2-L"
    assert bench_seats[-1] == "R4B3-R"

    # desk 3 rows × 4 cols = 12 seats
    desk_seats = all_seats(3, 4, 1, "desk")
    assert len(desk_seats) == 12
    assert desk_seats[0] == "R1C1"
    assert desk_seats[-1] == "R3C4"

    # round-trip
    lbl = "R2B3-M"
    parsed = parse_seat_label(lbl)
    assert seat_label(parsed["row"], parsed["bench"], parsed["pos"], "bench") == lbl

    print("seat_naming self-check: all OK")
    print("Sample bench layout (R1):", seats_per_row_labels(1, 3, 2, "bench"))
    print("Sample desk layout (R1):", seats_per_row_labels(1, 4, 1, "desk"))
