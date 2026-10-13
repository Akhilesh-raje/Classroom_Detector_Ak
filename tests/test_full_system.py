"""
SmartClass AI — Full system test.
Run: python tests/test_full_system.py
Exits 0 on all pass, 1 on any failure.
"""
import sys, os, io, json, sqlite3, tempfile, traceback, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PASS = []; FAIL = []

def ok(name):
    PASS.append(name); print(f'  [PASS] {name}')

def fail(name, reason=''):
    FAIL.append(name); print(f'  [FAIL] {name}  →  {reason}')

def section(title):
    print(f'\n── {title} {"─"*(55-len(title))}')


# ══════════════════════════════════════════════════════════════════════════════
# 1. SEAT NAMING
# ══════════════════════════════════════════════════════════════════════════════
section('1. SEAT NAMING')
try:
    from core.seat_naming import seat_label, parse_seat_label, all_seats, seats_per_row_labels, _POSITIONS

    # bench 2-seat
    assert seat_label(1,2,'L','bench') == 'R1B2-L'; ok('bench label R1B2-L')
    assert seat_label(2,3,'R','bench') == 'R2B3-R'; ok('bench label R2B3-R')
    assert seat_label(1,1,'M','bench') == 'R1B1-M'; ok('bench label R1B1-M')

    # desk
    assert seat_label(3,4,None,'desk') == 'R3C4';   ok('desk label R3C4')

    # round-trips
    for lbl in ['R1B2-L','R2B1-R','R3B3-M','R1C1','R4C4']:
        p = parse_seat_label(lbl)
        reconstructed = seat_label(p['row'], p['bench'], p['pos'], p['layout_type'])
        assert reconstructed == lbl, f'{reconstructed} != {lbl}'
    ok('all round-trips')

    # all_seats counts
    b = all_seats(4, 3, 2, 'bench')
    assert len(b) == 24, f'expected 24 got {len(b)}'
    assert b[0] == 'R1B1-L' and b[-1] == 'R4B3-R'; ok('bench all_seats count+order')

    d = all_seats(3, 4, 1, 'desk')
    assert len(d) == 12 and d[0]=='R1C1' and d[-1]=='R3C4'; ok('desk all_seats')

    # 3-seat bench
    t = all_seats(2, 2, 3, 'bench')
    assert len(t) == 12; ok('3-seat bench all_seats')

    # unknown label
    p = parse_seat_label('GARBAGE')
    assert p['layout_type'] == 'unknown'; ok('parse unknown label returns unknown')

    # edge: 1-seat per bench = desk-like
    s = all_seats(1, 1, 1, 'bench')
    assert len(s) == 1 and s[0] == 'R1B1-C'; ok('1-seat bench produces -C')

except Exception as e:
    fail('seat_naming', traceback.format_exc(limit=3))


# ══════════════════════════════════════════════════════════════════════════════
# 2. FACE RECOGNITION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
section('2. FACE RECOGNITION ENGINE')
try:
    import numpy as np
    from core.face_recognition_engine import FaceRecognitionEngine

    fre = FaceRecognitionEngine()

    # load_from_bytes stores correctly
    enc = np.random.rand(192).astype(np.float32)
    fre.load_from_bytes(10, 'Bob', 'R010', enc.tobytes())
    assert len(fre.registered) == 1; ok('load_from_bytes stores entry')

    # idempotent re-register (no duplicate)
    fre.load_from_bytes(10, 'Bob', 'R010', enc.tobytes())
    fre.load_from_bytes(10, 'Bob', 'R010', enc.tobytes())
    assert len(fre.registered) == 1; ok('re-register does not duplicate')

    # second student
    enc2 = np.random.rand(192).astype(np.float32)
    fre.load_from_bytes(11, 'Carol', 'R011', enc2.tobytes())
    assert len(fre.registered) == 2; ok('two students stored')

    # match exact
    mid, name, roll, conf = fre.match(enc)
    assert mid == 10 and name == 'Bob' and conf >= 0.99; ok('match exact encoding')

    # no match when empty
    fre2 = FaceRecognitionEngine()
    mid2, _, _, conf2 = fre2.match(enc)
    assert mid2 is None; ok('match returns None when no registered faces')

    # extract_embedding doesn't crash on garbage image
    import cv2
    noise = np.random.randint(0,255,(48,48,3),dtype=np.uint8)
    result = fre.extract_embedding(noise)
    assert result is None or isinstance(result, np.ndarray); ok('extract_embedding on noise no crash')

    # None input
    assert fre.extract_embedding(None) is None; ok('extract_embedding(None) = None')
    assert fre.extract_encoding(None) is None; ok('extract_encoding(None) = None')

    # bytes_to_encoding round-trip
    back = fre.bytes_to_encoding(enc.tobytes())
    assert np.allclose(back, enc, atol=1e-6); ok('bytes_to_encoding round-trip')

    # compare: identical = 1.0
    sim = fre.compare(enc, enc)
    assert abs(sim - 1.0) < 1e-5; ok('compare identical embeddings = 1.0')

    # compare: orthogonal ≈ 0
    e1 = np.zeros(4, dtype=np.float32); e1[0] = 1.0
    e2 = np.zeros(4, dtype=np.float32); e2[1] = 1.0
    assert abs(fre.compare(e1, e2)) < 1e-5; ok('compare orthogonal embeddings = 0')

except Exception as e:
    fail('face_recognition_engine', traceback.format_exc(limit=4))


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATABASE MIGRATION
# ══════════════════════════════════════════════════════════════════════════════
section('3. DATABASE MIGRATION')
try:
    from core.database import init_db, Base, engine, Student, SeatingConfig
    import sqlalchemy

    init_db()  # idempotent

    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__),'..','backend','smartclass.db'))
    cur  = conn.cursor()

    # All tables present
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {r[0] for r in cur.fetchall()}
    for t in ['students','attendance','activity_logs','sessions','session_reports','seating_configs']:
        if t in tables: ok(f'table {t} exists')
        else: fail(f'table {t} exists', 'MISSING')

    # Student has new columns
    cur.execute('PRAGMA table_info(students)')
    student_cols = {r[1] for r in cur.fetchall()}
    for c in ['seat_row','seat_bench','seat_pos','grid_cell','avg_attentiveness']:
        if c in student_cols: ok(f'students.{c} exists')
        else: fail(f'students.{c} exists', 'MISSING')

    # SeatingConfig has expected columns
    cur.execute('PRAGMA table_info(seating_configs)')
    sc_cols = {r[1] for r in cur.fetchall()}
    for c in ['layout_type','num_rows','benches_per_row','seats_per_bench','is_active']:
        if c in sc_cols: ok(f'seating_configs.{c} exists')
        else: fail(f'seating_configs.{c} exists', 'MISSING')

    conn.close()

except Exception as e:
    fail('database_migration', traceback.format_exc(limit=4))


# ══════════════════════════════════════════════════════════════════════════════
# 4. BACKEND API (in-process, no server needed)
# ══════════════════════════════════════════════════════════════════════════════
section('4. BACKEND API (in-process)')
try:
    from fastapi.testclient import TestClient
    import main as app_module
    client = TestClient(app_module.app)

    # Health-check: students list (empty)
    r = client.get('/api/students')
    assert r.status_code == 200
    assert isinstance(r.json(), list); ok('GET /api/students returns list')

    # Seating config default
    r = client.get('/api/seating-config')
    assert r.status_code == 200
    body = r.json()
    assert 'seats' in body and 'layout_type' in body; ok('GET /api/seating-config returns layout')
    assert isinstance(body['seats'], list) and len(body['seats']) > 0; ok('seating-config has seats list')

    # Save bench config
    r = client.post('/api/seating-config', data={
        'name':'Room 101','layout_type':'bench',
        'num_rows':'3','benches_per_row':'3','seats_per_bench':'2'
    })
    assert r.status_code == 200
    b = r.json()
    assert b['layout_type'] == 'bench' and b['seats_per_bench'] == 2
    assert len(b['seats']) == 18; ok('POST /api/seating-config bench 3x3x2=18 seats')

    # Save desk config
    r = client.post('/api/seating-config', data={
        'name':'Room 102','layout_type':'desk',
        'num_rows':'2','benches_per_row':'4','seats_per_bench':'1'
    })
    assert r.status_code == 200
    b = r.json()
    assert len(b['seats']) == 8; ok('POST /api/seating-config desk 2x4=8 seats')

    # Bad layout_type
    r = client.post('/api/seating-config', data={'layout_type':'pod','num_rows':'1','benches_per_row':'1','seats_per_bench':'1'})
    assert r.status_code == 422; ok('POST /api/seating-config invalid layout_type → 422')

    # Register a student
    r = client.post('/api/students', data={
        'name':'Test Student','rollNo':'TS001','section':'10-A','email':''
    })
    assert r.status_code == 200
    sid = r.json()['id']; ok(f'POST /api/students creates student id={sid}')

    # Duplicate roll number
    r2 = client.post('/api/students', data={
        'name':'Dup','rollNo':'TS001','section':'10-A'
    })
    assert r2.status_code == 400; ok('duplicate rollNo → 400')

    # Assign bench seat
    r = client.post(f'/api/students/{sid}/assign-seat', data={'grid_cell':'R1B2-L'})
    assert r.status_code == 200; ok('assign-seat bench format R1B2-L → 200')

    # Assign desk seat
    r = client.post(f'/api/students/{sid}/assign-seat', data={'grid_cell':'R2C3'})
    assert r.status_code == 200; ok('assign-seat desk format R2C3 → 200')

    # Bad seat format
    r = client.post(f'/api/students/{sid}/assign-seat', data={'grid_cell':'GARBAGE'})
    assert r.status_code == 422; ok('assign-seat bad format → 422')

    # Student profile (no sessions yet)
    r = client.get(f'/api/students/{sid}/profile')
    assert r.status_code == 200
    p = r.json()
    assert 'student' in p and 'attendance' in p and 'performance' in p; ok('GET /api/students/{id}/profile structure')
    assert p['performance']['trend'] == []; ok('profile trend empty when no sessions')
    assert p['attendance']['total_present'] == 0; ok('profile attendance=0 when no records')

    # Profile 404
    r = client.get('/api/students/99999/profile')
    assert r.status_code == 404; ok('profile nonexistent student → 404')

    # Face recognition status
    r = client.get('/api/face-recognition/status')
    assert r.status_code == 200
    assert 'backend' in r.json(); ok('GET /api/face-recognition/status')

    # Sessions list
    r = client.get('/api/sessions')
    assert r.status_code == 200; ok('GET /api/sessions')

    # Analytics summary
    r = client.get('/api/analytics/summary')
    assert r.status_code == 200
    assert 'totalStudents' in r.json(); ok('GET /api/analytics/summary')

    # Attendance
    r = client.get('/api/attendance')
    assert r.status_code == 200
    assert 'records' in r.json(); ok('GET /api/attendance')

    # Clean up test student
    client.delete(f'/api/students/{sid}')
    r = client.get('/api/students')
    assert all(s['id'] != sid for s in r.json()); ok('DELETE /api/students/{id}')

except ImportError:
    fail('api_in_process', 'httpx not installed — run: pip install httpx')
except Exception as e:
    fail('api_in_process', traceback.format_exc(limit=5))


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCAN_CLASSROOM FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
section('5. SCAN_CLASSROOM generate_seat_names + build_layout_json')
try:
    # Import only the two pure functions we need, bypassing the camera/cv2 setup
    # by extracting them directly from the source with exec into a controlled namespace
    sc_path = os.path.join(os.path.dirname(__file__), '..', 'scan_classroom.py')
    sc_path = os.path.normpath(sc_path)
    sc_ns = {'__file__': sc_path, '__name__': 'scan_classroom'}

    # Stub heavy imports so the module-level code doesn't crash
    import types, unittest.mock as mock
    fake_cv2 = types.ModuleType('cv2'); fake_cv2.CAP_DSHOW = 0
    stub_mods = ['cv2','ultralytics']
    originals = {}
    for m in stub_mods:
        originals[m] = sys.modules.get(m)
        if m not in sys.modules:
            sys.modules[m] = types.ModuleType(m)

    with open(sc_path, encoding='utf-8') as fh:
        src = fh.read()
    exec(compile(src, sc_path, 'exec'), sc_ns)

    # Restore stubs
    for m, orig in originals.items():
        if orig is None:
            sys.modules.pop(m, None)
        else:
            sys.modules[m] = orig

    generate_seat_names  = sc_ns['generate_seat_names']
    build_layout_json    = sc_ns['build_layout_json']

    # --- 2-seat bench ---
    rows_2 = [
        [{'bbox':[0,0,100,50],'center':[50,25],'cls_name':'bench','confidence':0.8,'area':5000}],
        [{'bbox':[0,60,100,110],'center':[50,85],'cls_name':'bench','confidence':0.8,'area':5000},
         {'bbox':[110,60,210,110],'center':[160,85],'cls_name':'bench','confidence':0.7,'area':5000}],
    ]
    generate_seat_names(rows_2, seats_per_bench=2)
    labels_2 = [d['seat_label'] for row in rows_2 for d in row]
    assert labels_2[0] == 'R1B1-L', f'got {labels_2[0]}'
    assert labels_2[1] == 'R2B1-L', f'got {labels_2[1]}'
    ok('generate_seat_names bench-2 correct labels')
    all_bench_labels = [l for row in rows_2 for d in row for l in d.get('bench_labels',[])]
    assert 'R1B1-R' in all_bench_labels, f'R1B1-R missing from {all_bench_labels}'
    ok('generate_seat_names bench-2 sub-labels include R')

    # --- 3-seat bench ---
    rows_3 = [[{'bbox':[0,0,150,50],'center':[75,25],'cls_name':'bench','confidence':0.8,'area':7500}]]
    generate_seat_names(rows_3, seats_per_bench=3)
    bench_labels_3 = rows_3[0][0].get('bench_labels',[])
    assert bench_labels_3 == ['R1B1-L','R1B1-M','R1B1-R'], f'got {bench_labels_3}'
    ok('generate_seat_names bench-3 L/M/R correct')

    # --- desk (1-seat) ---
    rows_d = [[{'bbox':[0,0,80,60],'center':[40,30],'cls_name':'desk','confidence':0.9,'area':4800},
               {'bbox':[90,0,170,60],'center':[130,30],'cls_name':'desk','confidence':0.9,'area':4800}]]
    generate_seat_names(rows_d, seats_per_bench=1)
    labels_d = [d['seat_label'] for row in rows_d for d in row]
    assert labels_d == ['R1C1','R1C2'], f'got {labels_d}'
    ok('generate_seat_names desk R1C1/R1C2 correct')

    # build_layout_json bench
    layout_b = build_layout_json(rows_2, 1920, 1080, 0, '2026-09-21', seats_per_bench=2)
    assert layout_b['version'] == '3.0'; ok('layout JSON version 3.0')
    assert layout_b['layout_type'] == 'bench'; ok('layout_type=bench stored')
    # 1 bench in row1 → 2 sub-seats, 2 benches in row2 → 4 sub-seats = 6 total
    assert layout_b['total_seats'] == 6, f'expected 6 got {layout_b["total_seats"]}'
    ok('bench layout seat count expands sub-seats correctly')
    assert 'R1B1-L' in layout_b['grid_cells']; ok('R1B1-L in grid_cells')
    assert 'R1B1-R' in layout_b['grid_cells']; ok('R1B1-R in grid_cells')

    # build_layout_json desk
    layout_d = build_layout_json(rows_d, 1280, 720, 0, '2026-09-21', seats_per_bench=1)
    assert layout_d['layout_type'] == 'desk'; ok('desk layout_type stored')
    assert layout_d['total_seats'] == 2; ok('desk seat count correct')

except Exception as e:
    fail('scan_classroom', traceback.format_exc(limit=5))


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 60)
total = len(PASS) + len(FAIL)
print(f'  TOTAL: {total}  |  PASS: {len(PASS)}  |  FAIL: {len(FAIL)}')
if FAIL:
    print(f'\n  FAILED:')
    for f in FAIL:
        print(f'    ✗  {f}')
    print()
    print('  RESULT: FAILURES — fix required')
    sys.exit(1)
else:
    print()
    print('  RESULT: ALL PASS ✓')
    sys.exit(0)
