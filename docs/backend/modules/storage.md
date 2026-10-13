# 💾 Storage & Database Subsystem (`backend/core/storage/`)

The Storage subsystem manages relational data persistence via **SQLAlchemy ORM** over **SQLite 3** (`smartclass.db`). It provides transactional consistency, automatic column migrations, and thread-safe session scoping.

---

## 🗄️ Relational Schema & Models

```mermaid
erDiagram
    SEATING_CONFIG ||--o{ STUDENT : assigns
    STUDENT ||--o{ ATTENDANCE : logs
    STUDENT ||--o{ ACTIVITY_LOG : records
    SESSION ||--o{ ATTENDANCE : tracks
    SESSION ||--o{ SESSION_REPORT : generates
    STUDENT ||--o{ SESSION_REPORT : summarizes

    STUDENT {
        int id PK
        string roll_no UK
        string name
        string section
        string email
        blob face_encoding
        text face_image
        string grid_cell
        int seat_row
        int seat_bench
        string seat_pos
        int total_attendance
        float avg_attentiveness
    }

    ATTENDANCE {
        int id PK
        int student_id FK
        string session_id FK
        string seat_id
        string date
        datetime marked_at
        float confidence
        string status
        string presence_state
        int detection_count
        int candidate_student_id
        text audit_trail
        int is_dry_run
    }

    ACTIVITY_LOG {
        int id PK
        int student_id FK
        datetime timestamp
        string activity
        float attention_score
        float head_yaw
        float head_pitch
        float eye_openness
    }

    SESSION {
        string session_id PK
        string session_name
        datetime start_time
        datetime end_time
        string status
        int total_frames
        int student_count
        float avg_class_attention
    }

    SESSION_REPORT {
        int id PK
        string session_id FK
        int student_id FK
        text report_json
        datetime created_at
    }

    SEATING_CONFIG {
        int id PK
        string name
        string layout_type
        int num_rows
        int benches_per_row
        int seats_per_bench
        int is_active
    }
```

---

## 🔍 Model Reference

### 1. `Student`
Represents an enrolled student in the institution:
- `face_encoding`: Raw binary bytes of normalized 512-D float32 vector (`numpy.ndarray.tobytes()`).
- `face_image`: Base64-encoded thumbnail for frontend avatar rendering.
- `grid_cell`: Primary registered seat location (e.g. `R1C1` or `R1B2-L`).

### 2. `AttendanceRecord`
Enterprise attendance ledger with verification tiers:
- `status`: Verification outcome (`VERIFIED`, `PENDING_REVIEW`, `REJECTED`, `MANUAL`).
- `presence_state`: Biometric state (`DETECTED`, `CANDIDATE`, `IDENTIFIED`, `UNKNOWN_OCCUPANT`).
- `audit_trail`: JSON evidence string preserving bounding boxes, matching distances, and frame indices.
- `is_dry_run`: `1` if generated during a simulated test, isolating real attendance scores.

### 3. `Session` & `SessionReport`
Audit trail of classroom lecture sessions:
- `session_id`: Unique UUIDv4.
- `report_json`: Structured JSON containing per-student timelines, distraction events, and attention distributions.

---

## 🛠️ Automatic Migrations (`_migrate_add_columns`)

SQLAlchemy's `Base.metadata.create_all()` does not alter existing tables. `init_db()` runs `_migrate_add_columns()`, which queries SQLite's `PRAGMA table_info(<table>)` and applies non-destructive `ALTER TABLE ADD COLUMN` queries dynamically at startup.
