from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
import time
import base64
import numpy as np
import database
from face_recognition_module import FaceRecognitionSystem, base64_to_image, detect_faces

app = Flask(__name__)

# Initialize DB on start
database.init_db()

# Initialize Face Recognition System
face_system = FaceRecognitionSystem()

# Drone Camera Stream Configuration
DRONE_STREAM_URL = os.environ.get("DRONE_STREAM_URL", "0")  # Default to device video index or RTSP/HTTP URL
drone_cap = None

def get_drone_cap():
    global drone_cap, DRONE_STREAM_URL
    if drone_cap is not None and drone_cap.isOpened():
        return drone_cap
    source = DRONE_STREAM_URL
    if source.isdigit():
        source = int(source)
    drone_cap = cv2.VideoCapture(source)
    return drone_cap

def generate_drone_frames():
    """Generates MJPEG stream frames from drone camera stream source."""
    while True:
        cap = get_drone_cap()
        if cap is None or not cap.isOpened():
            # If camera opened failed, generate placeholder frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Drone Camera Offline / Reconnecting...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            time.sleep(1.0)
        else:
            success, frame = cap.read()
            if not success or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No Frame Received from Drone Stream", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                time.sleep(0.5)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Returns video stream feed from drone camera."""
    return Response(generate_drone_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/drone/config', methods=['GET', 'POST'])
def drone_config():
    """Gets or sets drone stream source URL/device ID."""
    global DRONE_STREAM_URL, drone_cap
    if request.method == 'POST':
        data = request.get_json() or {}
        new_source = data.get('stream_url', '').strip()
        if new_source:
            DRONE_STREAM_URL = new_source
            if drone_cap is not None:
                drone_cap.release()
                drone_cap = None
            return jsonify({'success': True, 'message': f'Drone stream URL updated to {DRONE_STREAM_URL}', 'stream_url': DRONE_STREAM_URL})
        return jsonify({'success': False, 'message': 'Invalid stream URL'}), 400

    return jsonify({'success': True, 'stream_url': DRONE_STREAM_URL})

@app.route('/api/drone/snapshot', methods=['GET'])
def drone_snapshot():
    """Captures single snapshot from current drone stream and returns base64 image string."""
    cap = get_drone_cap()
    if cap is None or not cap.isOpened():
        return jsonify({'success': False, 'message': 'Drone camera stream unavailable'}), 503

    success, frame = cap.read()
    if not success or frame is None:
        return jsonify({'success': False, 'message': 'Failed to capture frame from drone stream'}), 500

    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return jsonify({'success': False, 'message': 'Failed to encode captured frame'}), 500

    b64_str = base64.b64encode(buffer).decode('utf-8')
    image_data_url = f"data:image/jpeg;base64,{b64_str}"
    return jsonify({'success': True, 'image': image_data_url})

@app.route('/api/register', methods=['POST'])
def register_student():
    """
    Expects JSON:
    {
       "name": "Student Name",
       "reg_no": "REG1234",
       "image": "data:image/jpeg;base64,..."
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No JSON payload provided'}), 400

    name = data.get('name', '').strip()
    reg_no = data.get('reg_no', '').strip()
    image_b64 = data.get('image', '')

    if not name or not reg_no:
        return jsonify({'success': False, 'message': 'Name and Registration Number are required.'}), 400

    if not image_b64:
        return jsonify({'success': False, 'message': 'Image is required for face registration.'}), 400

    img = base64_to_image(image_b64)
    if img is None:
        return jsonify({'success': False, 'message': 'Invalid image format.'}), 400

    faces, gray = detect_faces(img)
    if len(faces) == 0:
        return jsonify({'success': False, 'message': 'No face detected in the captured image. Please align face clearly.'}), 400

    # Get largest detected face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    gray_face = gray[y:y+h, x:x+w]

    try:
        student_id = database.add_student(reg_no, name)
        # Save face image and train/update model
        face_system.save_student_face(student_id, gray_face)
        return jsonify({
            'success': True,
            'message': f'Student "{name}" ({reg_no}) registered successfully.',
            'student': {'id': student_id, 'name': name, 'reg_no': reg_no}
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error registering student: {str(e)}'}), 500

@app.route('/api/attendance', methods=['POST'])
def mark_attendance():
    """
    Expects JSON:
    {
       "image": "data:image/jpeg;base64,..."
    }
    Performs ONLY facial recognition to identify student and record attendance.
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'success': False, 'message': 'Image is required for attendance.'}), 400

    image_b64 = data.get('image')
    result = face_system.recognize_face_from_image(image_b64)

    if result['status'] == 'no_face':
        return jsonify({
            'success': False,
            'registered': False,
            'message': 'No face detected. Please position your face clearly in front of the camera.'
        }), 400

    if result['status'] == 'not_registered' or result['student_id'] is None:
        return jsonify({
            'success': False,
            'registered': False,
            'message': 'Student is not registered.'
        }), 200

    student_id = result['student_id']
    student = database.get_student_by_id(student_id)

    if not student:
        return jsonify({
            'success': False,
            'registered': False,
            'message': 'Student is not registered.'
        }), 200

    attendance_record = database.mark_attendance(student_id)

    if attendance_record.get('already_marked'):
        msg = f'Attendance already marked today for {student["name"]} ({student["reg_no"]}).'
    else:
        msg = f'Attendance marked successfully for {student["name"]} ({student["reg_no"]}).'

    return jsonify({
        'success': True,
        'registered': True,
        'student': student,
        'attendance': attendance_record,
        'message': msg
    }), 200

@app.route('/api/students', methods=['GET'])
def get_students():
    students = database.get_all_students()
    return jsonify({'success': True, 'students': students})

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    logs = database.get_attendance_logs()
    return jsonify({'success': True, 'attendance': logs})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
