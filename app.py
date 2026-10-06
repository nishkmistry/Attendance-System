from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
import time
import base64
import numpy as np
import database
from face_recognition_module import (
    FaceRecognitionSystem, base64_to_image, image_to_base64,
    detect_faces_dnn, crop_circular_face
)

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

@app.route('/api/detect_face', methods=['POST'])
def detect_face_endpoint():
    """
    Expects JSON: { "image": "data:image/jpeg;base64,..." }
    Detects faces automatically and returns list of face bounding boxes and cropped circular face base64.
    """
    data = request.get_json() or {}
    image_b64 = data.get('image', '')
    if not image_b64:
        return jsonify({'success': False, 'message': 'Image is required.'}), 400

    img = base64_to_image(image_b64)
    if img is None:
        return jsonify({'success': False, 'message': 'Invalid image format.'}), 400

    detections = detect_faces_dnn(img)
    if len(detections) == 0:
        return jsonify({'success': True, 'detected': False, 'faces': [], 'message': 'No face detected'}), 200

    formatted_faces = []
    circular_face_b64 = None

    largest = max(detections, key=lambda d: d['box'][2] * d['box'][3])
    lx, ly, lw, lh = largest['box']
    crop, face_box = crop_circular_face(img, face_box=largest['box'], size=200, circular=True)
    if crop is not None:
        circular_face_b64 = image_to_base64(crop)

    for det in detections:
        x, y, w, h = det['box']
        formatted_faces.append({
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'cx': int(x + w / 2),
            'cy': int(y + h / 2),
            'r': int(max(w, h) / 2)
        })

    return jsonify({
        'success': True,
        'detected': True,
        'faces': formatted_faces,
        'primary_face': {
            'x': int(lx),
            'y': int(ly),
            'w': int(lw),
            'h': int(lh)
        },
        'cropped_face': circular_face_b64
    })

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

    existing = database.get_student_by_reg_no(reg_no)
    if existing:
        return jsonify({
            'success': False,
            'message': f'Registration Number "{reg_no}" is already registered to {existing["name"]}.'
        }), 409

    img = base64_to_image(image_b64)
    if img is None:
        return jsonify({'success': False, 'message': 'Invalid image format.'}), 400

    detections = detect_faces_dnn(img)
    if len(detections) == 0:
        return jsonify({'success': False, 'message': 'No face detected in the captured image. Please align face clearly.'}), 400

    # Get largest detected face
    largest = max(detections, key=lambda d: d['box'][2] * d['box'][3])
    x, y, w, h = largest['box']

    # Circular crop is for the UI preview only - never used for the recognition embedding
    circular_face, _ = crop_circular_face(img, face_box=largest['box'], size=200, circular=True)

    try:
        student_id = database.add_student(reg_no, name)
        # Save the rectangular aligned face crop and cache its embedding
        face_system.save_student_face(student_id, largest['face'])

        cropped_preview_b64 = image_to_base64(circular_face) if circular_face is not None else None
        return jsonify({
            'success': True,
            'message': f'Student "{name}" ({reg_no}) registered successfully.',
            'student': {'id': student_id, 'name': name, 'reg_no': reg_no},
            'face_box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
            'cropped_face': cropped_preview_b64
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
    Detects EVERY face in the frame (e.g. a drone shot of a classroom) in one pass.
    Registered faces are matched and marked present automatically; unmatched faces
    are labeled Unknown1, Unknown2, ... in detection order (resets every scan).
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'success': False, 'message': 'Image is required for attendance.'}), 400

    image_b64 = data.get('image')
    img = base64_to_image(image_b64)
    if img is None:
        return jsonify({'success': False, 'message': 'Invalid image format.'}), 400

    recognitions = face_system.recognize_faces_batch(img)

    if len(recognitions) == 0:
        return jsonify({
            'success': True,
            'faces': [],
            'marked_count': 0,
            'unknown_count': 0,
            'message': 'No faces detected in the frame.'
        }), 200

    faces_result = []
    marked_count = 0
    unknown_count = 0

    for rec in recognitions:
        x, y, w, h = rec['box']
        box = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}

        student = database.get_student_by_id(rec['student_id']) if rec['student_id'] is not None else None

        if student:
            attendance_record = database.mark_attendance(student['id'])
            already = attendance_record.get('already_marked', False)
            if not already:
                marked_count += 1
            faces_result.append({
                'box': box,
                'status': 'already_marked' if already else 'marked',
                'name': student['name'],
                'reg_no': student['reg_no'],
                'timestamp': attendance_record.get('timestamp'),
                'distance': rec['distance']
            })
        else:
            unknown_count += 1
            faces_result.append({
                'box': box,
                'status': 'unknown',
                'name': f'Unknown{unknown_count}',
                'reg_no': None,
                'timestamp': None,
                'distance': rec['distance']
            })

    return jsonify({
        'success': True,
        'faces': faces_result,
        'marked_count': marked_count,
        'unknown_count': unknown_count,
        'message': f'{marked_count} student(s) marked present, {unknown_count} unknown face(s) detected.'
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
    # Force the detection/embedding models to load now rather than on the first request -
    # cold start can take 20-40s on CPU, which otherwise makes the first registration or
    # attendance scan look hung. debug=True below always spawns a reloader child process
    # to actually serve, so only that child (not the watcher parent) needs to warm up.
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        face_system.warmup()
    app.run(host='0.0.0.0', port=5000, debug=True)
