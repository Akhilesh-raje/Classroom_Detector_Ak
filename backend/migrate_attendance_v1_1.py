"""
Migration script for SmartClass AI Attendance v1.1:
Makes `attendance.student_id` nullable and converts existing sentinel 0 to NULL,
preserving referential integrity with foreign key support.
"""
import sqlite3
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "smartclass.db"))

def migrate():
    print(f"Migrating database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check current table schema
    cursor.execute("PRAGMA table_info(attendance)")
    columns = cursor.fetchall()
    student_id_col = next((c for c in columns if c[1] == "student_id"), None)

    # In SQLite table_info: col_id, name, type, notnull, dflt_value, pk
    if student_id_col and student_id_col[3] == 0:
        print("[OK] attendance.student_id is already nullable.")
    else:
        print("[INFO] Rebuilding attendance table to make student_id nullable...")
        cursor.execute("PRAGMA foreign_keys = OFF")
        cursor.execute("""
            CREATE TABLE attendance_v1_1 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                date VARCHAR(10) NOT NULL,
                marked_at DATETIME,
                confidence FLOAT,
                captured_image TEXT,
                method VARCHAR(30),
                session_id VARCHAR(36),
                seat_id VARCHAR(20),
                status VARCHAR(30) DEFAULT 'VERIFIED',
                presence_state VARCHAR(30) DEFAULT 'IDENTIFIED',
                detection_count INTEGER DEFAULT 1,
                candidate_student_id INTEGER,
                audit_trail TEXT,
                is_dry_run INTEGER DEFAULT 0,
                FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE SET NULL,
                FOREIGN KEY(candidate_student_id) REFERENCES students(id) ON DELETE SET NULL
            )
        """)
        cursor.execute("""
            INSERT INTO attendance_v1_1 (
                id, student_id, date, marked_at, confidence, captured_image, method,
                session_id, seat_id, status, presence_state, detection_count,
                candidate_student_id, audit_trail, is_dry_run
            )
            SELECT 
                id, 
                CASE WHEN student_id = 0 THEN NULL ELSE student_id END, 
                date, marked_at, confidence, captured_image, method,
                session_id, seat_id, status, presence_state, detection_count,
                candidate_student_id, audit_trail, is_dry_run
            FROM attendance
        """)
        cursor.execute("DROP TABLE attendance")
        cursor.execute("ALTER TABLE attendance_v1_1 RENAME TO attendance")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_attendance_date ON attendance(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_attendance_student_id ON attendance(student_id)")
        cursor.execute("PRAGMA foreign_keys = ON")
        conn.commit()
        print("[SUCCESS] Table migrated successfully! student_id is now nullable.")

    # Verify migration
    cursor.execute("PRAGMA table_info(attendance)")
    columns = cursor.fetchall()
    for col in columns:
        if col[1] in ("student_id", "candidate_student_id", "status", "presence_state"):
            print(f"  Column: {col[1]}, Type: {col[2]}, NotNull: {col[3]}")

    conn.close()

if __name__ == "__main__":
    migrate()
