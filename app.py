from flask import Flask, render_template, request, jsonify
import os
import cv2
import database
from face_recognition_module import FaceRecognitionSystem, base64_to_image, detect_faces

app = Flask(__name__)

# Initialize DB on start
database.init_db()

# Initialize Face Recognition System
face_system = FaceRecognitionSystem()

@app.route('/')
def index():
    return render_template('index.html')

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
        # Save face image and train model
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
