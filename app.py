from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
import time
import base64
import threading
import numpy as np
from werkzeug.utils import secure_filename
import database
from face_recognition_module import (
    FaceRecognitionSystem, base64_to_image, image_to_base64,
    detect_faces_dnn, crop_circular_face, face_quality, ENROLL_QUALITY
)

app = Flask(__name__)

# Initialize DB on start
database.init_db()

# Initialize Face Recognition System
face_system = FaceRecognitionSystem()

# Recorded videos uploaded from the UI to stand in for a live drone feed
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v')
SNAPSHOT_WAIT_S = 3.0


class DroneStream:
    """
    Reads the drone source on a single background thread and keeps only the newest frame.
    The MJPEG feed and snapshots both serve that frame, so the frame that gets scanned is
    the one on screen, and a VideoCapture is never read from two threads at once.

    A video file is played at its own frame rate and loops when it ends, so recorded
    footage can simulate a drone flight when no drone is available.
    """
    IDLE_TIMEOUT_S = 15.0  # stop reading once nobody has asked for a frame in this long

    def __init__(self, source):
        self.source = source
        self._cond = threading.Condition()
        self._frame = None
        self._thread = None
        self._last_access = 0.0

    @property
    def is_video_file(self):
        return os.path.isfile(self.source)

    def set_source(self, source):
        with self._cond:
            self.source = source
            self._thread = None  # the old reader notices it was replaced and exits
            self._frame = None

    def get_frame(self, wait=0.0):
        """Newest frame, or None if the source isn't delivering. Waits up to `wait` s for a first frame."""
        deadline = time.monotonic() + wait
        with self._cond:
            self._last_access = time.monotonic()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, args=(self.source,), daemon=True)
                self._thread.start()
            while self._frame is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            return self._frame

    def _owns(self):
        return self._thread is threading.current_thread()

    def _keep_running(self):
        with self._cond:
            if not self._owns():
                return False
            if time.monotonic() - self._last_access > self.IDLE_TIMEOUT_S:
                self._thread = None
                self._frame = None
                return False
            return True

    def _publish(self, frame):
        with self._cond:
            if self._owns():
                self._frame = frame
                self._cond.notify_all()

    def _run(self, source):
        is_file = os.path.isfile(source)
        cap = None
        try:
            while self._keep_running():
                if cap is None:
                    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        self._publish(None)
                        time.sleep(1.0)
                        continue
                    interval = 0  # live sources: read as fast as they deliver
                    if is_file:   # files: pace playback to real time
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        interval = 1.0 / fps if 1 <= fps <= 120 else 1 / 25
                    next_due = time.monotonic()
                    frames_read = 0

                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    cap = None
                    if is_file and frames_read > 0:
                        continue  # end of the clip: reopen it and play again from the start
                    self._publish(None)
                    time.sleep(1.0)
                    continue

                frames_read += 1
                self._publish(frame)
                if interval:
                    next_due += interval
                    delay = next_due - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_due = time.monotonic()  # fell behind: don't speed up to catch up
        finally:
            if cap is not None:
                cap.release()


# Drone Camera Stream Configuration
# Device video index, RTSP/HTTP URL, or path to a video file (simulation)
drone_stream = DroneStream(os.environ.get("DRONE_STREAM_URL", "0"))


def stream_info():
    return {'stream_url': drone_stream.source, 'mode': 'video' if drone_stream.is_video_file else 'live'}


def generate_drone_frames():
    """Generates MJPEG stream frames from drone camera stream source."""
    last = None
    while True:
        frame = drone_stream.get_frame()
        if frame is None:
            last = None
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Drone Camera Offline / Reconnecting...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            time.sleep(0.5)
        elif frame is last:
            time.sleep(0.01)  # nothing new yet - don't re-encode the same frame
            continue
        else:
            last = frame

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', model_name=face_system.model_name)

@app.route('/video_feed')
def video_feed():
    """Returns video stream feed from drone camera."""
    return Response(generate_drone_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/drone/config', methods=['GET', 'POST'])
def drone_config():
    """Gets or sets drone stream source URL/device ID/video file path."""
    if request.method == 'POST':
        data = request.get_json() or {}
        # Windows "Copy as path" wraps the path in quotes
        new_source = data.get('stream_url', '').strip().strip('"\'').strip()
        if new_source:
            drone_stream.set_source(new_source)
            return jsonify({'success': True, 'message': f'Drone stream URL updated to {new_source}', **stream_info()})
        return jsonify({'success': False, 'message': 'Invalid stream URL'}), 400

    return jsonify({'success': True, **stream_info()})

@app.route('/api/drone/upload', methods=['POST'])
def drone_upload():
    """
    Saves an uploaded video and plays it as the drone feed (looping), so the whole
    attendance pipeline can be demonstrated from recorded footage without a drone.
    """
    file = request.files.get('video')
    if file is None or not file.filename:
        return jsonify({'success': False, 'message': 'Choose a video file to upload.'}), 400

    stem, ext = os.path.splitext(file.filename)
    ext = ext.lower()
    if ext not in VIDEO_EXTENSIONS:
        return jsonify({'success': False, 'message': f'Unsupported file type. Use one of: {", ".join(VIDEO_EXTENSIONS)}'}), 400

    # Timestamped so a re-upload never overwrites a file the stream may still have open
    filename = f'{time.strftime("%Y%m%d-%H%M%S")}-{secure_filename(stem) or "video"}{ext}'
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    cap = cv2.VideoCapture(path)
    readable = cap.isOpened() and cap.read()[0]
    cap.release()
    if not readable:
        os.remove(path)
        return jsonify({'success': False, 'message': 'This video could not be decoded. Convert it to MP4 (H.264) and try again.'}), 400

    drone_stream.set_source(path)
    return jsonify({'success': True, 'message': f'Playing {file.filename} as the simulated drone feed.', **stream_info()})

@app.route('/api/drone/snapshot', methods=['GET'])
def drone_snapshot():
    """Captures single snapshot from current drone stream and returns base64 image string."""
    frame = drone_stream.get_frame(wait=SNAPSHOT_WAIT_S)
    if frame is None:
        return jsonify({'success': False, 'message': 'Drone camera stream unavailable'}), 503

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

MAX_ENROLL_FRAMES = 8

ENROLL_PROBLEMS = {
    'no_face': 'No face detected in the captured image. Please align face clearly.',
    'multiple_faces': 'More than one face is in view. Only the student being registered should be in frame.',
    'too_small': 'Face is too far from the camera. Move closer and try again.',
    'too_dark': 'Face is too dark. Improve the lighting (avoid a bright window behind the student) and try again.',
    'blurry': 'Face is too blurry. Hold still, face the camera, and try again.',
}


def _face_area(det):
    return det['box'][2] * det['box'][3]


@app.route('/api/register', methods=['POST'])
def register_student():
    """
    Expects JSON:
    {
       "name": "Student Name",
       "reg_no": "REG1234",
       "images": ["data:image/jpeg;base64,...", ...]   (or a single "image")
    }
    Several frames give several face samples per student, which makes recognition far
    more robust to pose/expression than a single snapshot. Registering an existing
    registration number again under the same name replaces that student's face samples.
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No JSON payload provided'}), 400

    name = data.get('name', '').strip()
    reg_no = data.get('reg_no', '').strip()
    images_b64 = [i for i in (data.get('images') or [data.get('image', '')]) if i][:MAX_ENROLL_FRAMES]

    if not name or not reg_no:
        return jsonify({'success': False, 'message': 'Name and Registration Number are required.'}), 400

    if not images_b64:
        return jsonify({'success': False, 'message': 'Image is required for face registration.'}), 400

    existing = database.get_student_by_reg_no(reg_no)
    if existing and existing['name'].strip().lower() != name.lower():
        return jsonify({
            'success': False,
            'message': f'Registration Number "{reg_no}" is already registered to {existing["name"]}.'
        }), 409

    # Keep the student's face from every frame that is clearly usable
    candidates = []  # (frame, detection)
    problem = None
    decoded_any = False
    for image_b64 in images_b64:
        img = base64_to_image(image_b64)
        if img is None:
            continue
        decoded_any = True
        detections = sorted(detect_faces_dnn(img), key=_face_area, reverse=True)
        if not detections:
            problem = problem or 'no_face'
            continue
        # A second face nearly as large as the first means we can't be sure which one
        # is the student - enrolling the wrong one would mark them for someone else.
        if len(detections) > 1 and _face_area(detections[1]) >= 0.5 * _face_area(detections[0]):
            problem = 'multiple_faces'
            continue
        reason = face_quality(detections[0]['face'], detections[0]['box'], **ENROLL_QUALITY)
        if reason:
            problem = reason
            continue
        candidates.append((img, detections[0]))

    if not decoded_any:
        return jsonify({'success': False, 'message': 'Invalid image format.'}), 400
    if not candidates:
        return jsonify({'success': False, 'message': ENROLL_PROBLEMS[problem or 'no_face']}), 400

    samples, conflict_id = face_system.build_enrollment(
        [det['aligned'] for _, det in candidates],
        exclude_student_id=existing['id'] if existing else None,
    )
    if not samples:
        return jsonify({'success': False, 'message': 'Could not read the face clearly. Please try again.'}), 400
    if conflict_id is not None:
        other = database.get_student_by_id(conflict_id)
        who = f'{other["name"]} ({other["reg_no"]})' if other else f'student #{conflict_id}'
        return jsonify({
            'success': False,
            'message': f'This face is already registered as {who}. '
                       f'If that is a different person, re-register {who} in better lighting first.'
        }), 409

    img, primary = candidates[0]
    x, y, w, h = primary['box']

    # Circular crop is for the UI preview only - never used for the recognition embedding
    circular_face, _ = crop_circular_face(img, face_box=primary['box'], size=200, circular=True)

    try:
        if existing:
            student_id = existing['id']
            face_system.replace_student_faces(student_id, samples)
            message = f'Face samples for "{existing["name"]}" ({reg_no}) updated ({len(samples)} captured).'
        else:
            student_id = database.add_student(reg_no, name)
            for crop, emb in samples:
                face_system.save_student_face(student_id, crop, embedding=emb)
            message = f'Student "{name}" ({reg_no}) registered successfully ({len(samples)} face samples).'

        cropped_preview_b64 = image_to_base64(circular_face) if circular_face is not None else None
        return jsonify({
            'success': True,
            'message': message,
            'student': {'id': student_id, 'name': existing['name'] if existing else name, 'reg_no': reg_no},
            'samples': len(samples),
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
                'distance': rec['distance'],
                'reason': rec.get('reason') or 'no_match'
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
    # Load the detection/embedding models (and rebuild a stale embeddings cache) now rather
    # than on the first request, which otherwise makes the first registration or attendance
    # scan look hung. debug=True below always spawns a reloader child process to actually
    # serve, so only that child (not the watcher parent) needs to warm up.
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        face_system.warmup()
    app.run(host='0.0.0.0', port=5000, debug=True)
