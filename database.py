import sqlite3
from datetime import datetime
import os

def get_db_path():
    return os.environ.get("DATABASE_PATH", "attendance.db")

def get_db_connection(db_path=None):
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=None):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reg_no TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            reg_no TEXT NOT NULL,
            name TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            date TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')

    conn.commit()
    conn.close()

def add_student(reg_no, name, db_path=None):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO students (reg_no, name) VALUES (?, ?)',
            (reg_no.strip(), name.strip())
        )
        conn.commit()
        student_id = cursor.lastrowid
        return student_id
    except sqlite3.IntegrityError:
        # Registration number already exists
        cursor.execute('SELECT id FROM students WHERE reg_no = ?', (reg_no.strip(),))
        row = cursor.fetchone()
        if row:
            # Update name if reg_no exists
            cursor.execute('UPDATE students SET name = ? WHERE id = ?', (name.strip(), row['id']))
            conn.commit()
            return row['id']
        raise
    finally:
        conn.close()

def get_student_by_id(student_id, db_path=None):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM students WHERE id = ?', (student_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_student_by_reg_no(reg_no, db_path=None):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM students WHERE reg_no = ?', (reg_no.strip(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_all_students(db_path=None):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM students ORDER BY name ASC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def mark_attendance(student_id, db_path=None):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Check student existence
    cursor.execute('SELECT * FROM students WHERE id = ?', (student_id,))
    student = cursor.fetchone()
    if not student:
        conn.close()
        return None

    today_date = datetime.now().strftime("%Y-%m-%d")

    # Check if already marked today
    cursor.execute(
        'SELECT * FROM attendance WHERE student_id = ? AND date = ?',
        (student_id, today_date)
    )
    existing = cursor.fetchone()

    if existing:
        conn.close()
        record = dict(existing)
        record['already_marked'] = True
        return record

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        'INSERT INTO attendance (student_id, reg_no, name, timestamp, date) VALUES (?, ?, ?, ?, ?)',
        (student['id'], student['reg_no'], student['name'], now_str, today_date)
    )
    conn.commit()
    att_id = cursor.lastrowid

    cursor.execute('SELECT * FROM attendance WHERE id = ?', (att_id,))
    new_record = dict(cursor.fetchone())
    new_record['already_marked'] = False
    conn.close()
    return new_record

def get_attendance_logs(db_path=None):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM attendance ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
