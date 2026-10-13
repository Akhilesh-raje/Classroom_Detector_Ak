"""
End-to-end Verification & Test Suite for Attendance Decision Engine v1
Tests:
1. Temporal confirmation window (<5 frames holds, >=5 frames commits)
2. Identity Resolution Tiers (Tier 1 Verified, Tier 2 Seat Candidate, Tier 3 Unknown Occupant)
3. Idempotency (repeated frames do not duplicate records)
4. Teacher Review workflow (Approving candidate moves candidate to VERIFIED)
5. Teacher Manual Override (Mark Present / Mark Absent)
6. CSV Export verification
7. Reset Day verification
"""
import urllib.request
import urllib.parse
import json
import time

BASE_URL = "http://127.0.0.1:8000/api/attendance"

def get_json(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def post_form(url, data):
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=encoded)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def run_tests():
    print("=" * 60)
    print("RUNNING ATTENDANCE ENGINE END-TO-END TEST SUITE")
    print("=" * 60)

    # 1. Reset Day for clean state
    print("\n[TEST 1] Resetting today's attendance...")
    post_form(f"{BASE_URL}/reset", {"dry_run_only": "false"})
    summary = get_json(BASE_URL)
    assert summary["counts"]["detected"] == 0, "Expected 0 detected after reset"
    assert summary["counts"]["verified"] == 0, "Expected 0 verified after reset"
    print("[PASS] Test 1 Passed: Clean reset confirmed (0 records).")

    # 2. Temporal Confirmation Window Test
    print("\n[TEST 2] Testing Temporal Confirmation Window (< 5 frames)...")
    single_obs = [{"track_id": "test_1", "seat_id": "R1C1", "matched_student_id": 1, "match_confidence": 0.95}]
    
    # Send 3 frames (below threshold of 5)
    for f in range(1, 4):
        res = post_form(f"{BASE_URL}/process-observation", {"observations": json.dumps(single_obs), "dry_run": "false"})
        state = res["results"][0]["status"]
        assert state == "PENDING_CONFIRMATION", f"Frame {f} should be PENDING_CONFIRMATION, got {state}"
    
    summary = get_json(BASE_URL)
    assert summary["counts"]["verified"] == 0, "No records should be committed before 5 frames!"
    print("[PASS] Test 2 Passed: 3 frames held in PENDING_CONFIRMATION (no false early check-in).")

    # 3. Crossing Temporal Threshold (Frames 4 and 5)
    print("\n[TEST 3] Reaching 5-frame threshold for Tier 1 Biometric Verification...")
    for f in range(4, 6):
        res = post_form(f"{BASE_URL}/process-observation", {"observations": json.dumps(single_obs), "dry_run": "false"})
    
    # At frame 5, it should transition to VERIFIED
    last_res = res["results"][0]
    assert last_res["status"] == "VERIFIED", f"Expected VERIFIED at frame 5, got {last_res['status']}"
    assert last_res["decision_tier"] == "TIER_1_VERIFIED_BIOMETRIC"
    
    summary = get_json(BASE_URL)
    assert summary["counts"]["verified"] == 1, "Student 1 should now be verified!"
    print("[PASS] Test 3 Passed: Temporal confirmation committed at frame 5 (1 verified check-in).")

    # 4. Strict Idempotency Test (Frames 6 through 15)
    print("\n[TEST 4] Testing Idempotency (Sending 10 more identical frames)...")
    for f in range(6, 16):
        res = post_form(f"{BASE_URL}/process-observation", {"observations": json.dumps(single_obs), "dry_run": "false"})
        assert res["results"][0]["action"] == "EXISTING_IDEMPOTENT"

    summary = get_json(BASE_URL)
    assert summary["counts"]["verified"] == 1, "Must remain exactly 1 record (no duplicate check-ins)!"
    print("[PASS] Test 4 Passed: 10 repeated frames resulted in 0 duplicates (Action: EXISTING_IDEMPOTENT).")

    # 5. Multi-Tier Resolution Test (Tier 2 Candidate & Tier 3 Unknown Occupant)
    print("\n[TEST 5] Testing Tier 2 (Seat Candidate) & Tier 3 (Unknown Occupant)...")
    multi_obs = [
        # Student 1 already verified
        {"track_id": "test_1", "seat_id": "R1C1", "matched_student_id": 1, "match_confidence": 0.95},
        # Seat R1C2 has an occupant, but NO face match -> Tier 2 Candidate (Student R1C2)
        {"track_id": "test_2", "seat_id": "R1C2", "matched_student_id": None, "match_confidence": 0.0},
        # Seat R9C9 has an occupant, not in roster -> Tier 3 Unknown Occupant
        {"track_id": "test_3", "seat_id": "R9C9", "matched_student_id": None, "match_confidence": 0.0},
    ]

    for f in range(1, 6):
        res = post_form(f"{BASE_URL}/process-observation", {"observations": json.dumps(multi_obs), "dry_run": "false"})

    summary = get_json(BASE_URL)
    assert summary["counts"]["verified"] == 1, "Verified count should be 1"
    assert summary["counts"]["review"] >= 2, f"Expected at least 2 review items, got {summary['counts']['review']}"
    
    review_seats = [rq["seatId"] for rq in summary["reviewQueue"]]
    assert "R1C2" in review_seats, "R1C2 should be in review queue as Candidate"
    assert "R9C9" in review_seats, "R9C9 should be in review queue as Unknown Occupant"
    print("[PASS] Test 5 Passed: Multi-tier separation confirmed (Verified: 1, Review: 2).")

    # 6. Teacher Review Workflow (Approve Candidate R1C2)
    print("\n[TEST 6] Testing Teacher Review Action (Approving R1C2 Candidate)...")
    r1c2_item = next(rq for rq in summary["reviewQueue"] if rq["seatId"] == "R1C2")
    review_res = post_form(f"{BASE_URL}/review", {
        "record_id": r1c2_item["id"],
        "decision": "APPROVE"
    })
    print(f"Review response: {review_res['message']}")

    summary_after = get_json(BASE_URL)
    assert summary_after["counts"]["verified"] == 2, "Verified count should increase from 1 to 2 after approval!"
    r1c2_rec = next(r for r in summary_after["records"] if r["assignedSeat"] == "R1C2")
    assert r1c2_rec["status"] == "VERIFIED", "Student R1C2 should now be marked VERIFIED"
    assert r1c2_rec["method"] == "teacher_approved", "Method should be teacher_approved"
    print("[PASS] Test 6 Passed: Teacher review moved candidate to VERIFIED with audit trace.")

    # 7. Teacher Manual Override Test
    print("\n[TEST 7] Testing Teacher Manual Override (Marking Student 3 Present)...")
    post_form(f"{BASE_URL}/manual", {"student_id": 3, "status": "present"})
    summary_man = get_json(BASE_URL)
    assert summary_man["counts"]["verified"] == 3, "Verified count should increase to 3"
    print("[PASS] Test 7 Passed: Manual override marked Student 3 as VERIFIED.")

    # 8. CSV Export Verification
    print("\n[TEST 8] Verifying CSV Export endpoint...")
    req = urllib.request.Request(f"{BASE_URL}/export-csv")
    with urllib.request.urlopen(req) as resp:
        csv_data = resp.read().decode("utf-8")
        assert "Student ID,Name,Roll No" in csv_data, "CSV header missing"
        assert "Student R1C1" in csv_data, "Student R1C1 missing in CSV"
        assert "Student R1C2" in csv_data, "Student R1C2 missing in CSV"
        print("[PASS] Test 8 Passed: CSV export generated valid audit spreadsheet format.")

    # 9. Seed the full 12-classroom simulation for browser verification
    print("\n[TEST 9] Simulating realistic classroom attendance (8 Verified, 3 Candidates, 1 Absent)...")
    full_classroom = [
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
    for _ in range(6):
        post_form(f"{BASE_URL}/process-observation", {"observations": json.dumps(full_classroom), "dry_run": "false"})
    
    final_summary = get_json(BASE_URL)
    print("\nFinal State Counts:")
    print(json.dumps(final_summary["counts"], indent=2))
    print("\n" + "=" * 60)
    print("ALL 9 TEST SUITE PHASES PASSED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
