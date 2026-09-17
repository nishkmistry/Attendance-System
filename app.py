from flask import Flask, render_template, request, jsonify
import os
import time
import cv2
import database
from face_recognition_module import FaceRecognitionSystem, base64_to_image, detect_faces
from web3_bridge import Web3Bridge

app = Flask(__name__)

# Initialize DB on start
database.init_db()

# Initialize Face Recognition System & Web3 Bridge
face_system = FaceRecognitionSystem()
web3_bridge = Web3Bridge()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/register', methods=['POST'])
def register_student():
    """
    Registers student, extracts 128-d vector embedding, and trains model.
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
        # Save face image, train model, and extract 128-d vector
        embedding = face_system.save_student_face(student_id, gray_face)
        database.save_student_embedding(student_id, embedding)

        return jsonify({
            'success': True,
            'message': f'Student "{name}" ({reg_no}) registered successfully with 128-d facial embedding.',
            'student': {'id': student_id, 'name': name, 'reg_no': reg_no}
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error registering student: {str(e)}'}), 500

@app.route('/api/attendance', methods=['POST'])
def mark_attendance():
    """
    Executes anti-spoofing check, facial vector matching, and Web3 smart contract dispatch.
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'success': False, 'message': 'Image is required for attendance.'}), 400

    image_b64 = data.get('image')

    # Candidate embeddings from DB
    candidate_embeddings = database.get_candidate_embeddings()

    start_inf = time.time()
    result = face_system.recognize_face_from_image(image_b64, candidate_embeddings=candidate_embeddings)
    inference_latency_ms = round((time.time() - start_inf) * 1000, 2)

    if result['status'] == 'no_face':
        return jsonify({
            'success': False,
            'registered': False,
            'message': 'No face detected. Please position your face clearly in front of the camera.'
        }), 400

    if result['status'] == 'spoof_detected':
        return jsonify({
            'success': False,
            'registered': False,
            'message': f"Anti-spoofing alert: {result['message']} (EAR: {result.get('ear', 0)})",
            'ear': result.get('ear', 0)
        }), 400

    if result['status'] == 'not_registered' or result['student_id'] is None:
        return jsonify({
            'success': False,
            'registered': False,
            'message': 'Student is not registered.',
            'confidence': result.get('confidence', 0)
        }), 200

    student_id = result['student_id']
    student = database.get_student_by_id(student_id)

    if not student:
        return jsonify({
            'success': False,
            'registered': False,
            'message': 'Student is not registered.'
        }), 200

    # Dispatch Web3 transaction to blockchain smart contract
    epoch_now = int(time.time())
    web3_res = web3_bridge.mark_attendance_on_chain(
        student_id=student['reg_no'],
        epoch_timestamp=epoch_now,
        location_id="TERMINAL_LOCATION_MAIN"
    )

    attendance_record = database.mark_attendance(
        student_id=student_id,
        web3_data=web3_res,
        inference_latency_ms=inference_latency_ms
    )

    if attendance_record.get('already_marked'):
        msg = f'Attendance already marked today for {student["name"]} ({student["reg_no"]}).'
    else:
        msg = f'Attendance marked & confirmed on-chain for {student["name"]} ({student["reg_no"]}).'

    explorer_link = f"https://sepolia.etherscan.io/tx/{web3_res['tx_hash']}" if web3_res.get('tx_hash') else None

    return jsonify({
        'success': True,
        'registered': True,
        'student': student,
        'attendance': attendance_record,
        'web3': web3_res,
        'explorer_link': explorer_link,
        'inference_latency_ms': inference_latency_ms,
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

@app.route('/api/web3/status', methods=['GET'])
def get_web3_status():
    status = web3_bridge.get_status()
    return jsonify({'success': True, 'web3_status': status})

@app.route('/api/benchmarks', methods=['GET'])
def get_benchmarks():
    metrics = database.get_benchmark_metrics()
    return jsonify({'success': True, 'benchmarks': metrics})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
