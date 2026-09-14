import unittest
from unittest.mock import patch
import os
import json
import numpy as np
import cv2
import base64
import database
from face_recognition_module import FaceRecognitionSystem
from app import app, face_system

class FacialAttendanceTestCase(unittest.TestCase):
    def setUp(self):
        self.test_db = 'test_attendance.db'
        self.test_dataset = 'test_dataset'
        self.test_model = 'test_trainer.yml'

        os.environ['DATABASE_PATH'] = self.test_db
        database.init_db(self.test_db)

        # Ensure fresh database tables for test isolation
        conn = database.get_db_connection(self.test_db)
        conn.execute('DELETE FROM attendance')
        conn.execute('DELETE FROM students')
        conn.commit()
        conn.close()

        # Override face_system settings for test isolation
        face_system.dataset_dir = self.test_dataset
        face_system.model_path = self.test_model
        face_system.load_model()

        app.config['TESTING'] = True
        self.client = app.test_client()

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        if os.path.exists(self.test_model):
            os.remove(self.test_model)
        if os.path.exists(self.test_dataset):
            import shutil
            shutil.rmtree(self.test_dataset)

    def _create_sample_image_b64(self):
        """Creates a sample dummy image."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        _, buffer = cv2.imencode('.jpg', img)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"

    def test_database_operations(self):
        # Add student
        student_id = database.add_student('REG001', 'Alice', db_path=self.test_db)
        self.assertIsNotNone(student_id)

        # Get student
        student = database.get_student_by_id(student_id, db_path=self.test_db)
        self.assertEqual(student['name'], 'Alice')
        self.assertEqual(student['reg_no'], 'REG001')

        # Mark attendance
        rec1 = database.mark_attendance(student_id, db_path=self.test_db)
        self.assertFalse(rec1['already_marked'])
        self.assertEqual(rec1['name'], 'Alice')

        # Duplicate attendance on same date
        rec2 = database.mark_attendance(student_id, db_path=self.test_db)
        self.assertTrue(rec2['already_marked'])

        # Fetch logs
        logs = database.get_attendance_logs(db_path=self.test_db)
        self.assertEqual(len(logs), 1)

    @patch('app.detect_faces')
    def test_register_student_no_face(self, mock_detect_faces):
        mock_detect_faces.return_value = ([], None)
        sample_img_b64 = self._create_sample_image_b64()

        response = self.client.post('/api/register', json={
            'name': 'Bob',
            'reg_no': 'REG002',
            'image': sample_img_b64
        })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('No face detected', data['message'])

    @patch('face_recognition_module.detect_faces')
    @patch('app.detect_faces')
    def test_register_and_mark_attendance_flow(self, mock_app_detect, mock_mod_detect):
        # Mock face detection returning 1 face box (10, 10, 50, 50) and gray image
        dummy_gray = np.ones((100, 100), dtype=np.uint8) * 100
        cv2.rectangle(dummy_gray, (20, 20), (40, 40), 200, -1)
        mock_faces = ([(10, 10, 50, 50)], dummy_gray)

        mock_app_detect.return_value = mock_faces
        mock_mod_detect.return_value = mock_faces

        face_img_b64 = self._create_sample_image_b64()

        # 1. Register student
        reg_response = self.client.post('/api/register', json={
            'name': 'Charlie',
            'reg_no': 'REG003',
            'image': face_img_b64
        })
        self.assertEqual(reg_response.status_code, 200)
        reg_data = json.loads(reg_response.data)
        self.assertTrue(reg_data['success'])

        # 2. Mark attendance using the same facial image
        att_response = self.client.post('/api/attendance', json={
            'image': face_img_b64
        })
        self.assertEqual(att_response.status_code, 200)
        att_data = json.loads(att_response.data)
        self.assertTrue(att_data['success'])
        self.assertTrue(att_data['registered'])
        self.assertEqual(att_data['student']['name'], 'Charlie')
        self.assertEqual(att_data['student']['reg_no'], 'REG003')

    @patch('face_recognition_module.detect_faces')
    def test_unregistered_student_attendance(self, mock_mod_detect):
        dummy_gray = np.ones((100, 100), dtype=np.uint8) * 100
        mock_faces = ([(10, 10, 50, 50)], dummy_gray)
        mock_mod_detect.return_value = mock_faces

        sample_img_b64 = self._create_sample_image_b64()
        att_response = self.client.post('/api/attendance', json={
            'image': sample_img_b64
        })
        self.assertEqual(att_response.status_code, 200)
        att_data = json.loads(att_response.data)
        self.assertFalse(att_data['success'])
        self.assertFalse(att_data['registered'])
        self.assertIn('Student is not registered', att_data['message'])

    def test_get_students_and_attendance_endpoints(self):
        # Fetch empty students list
        res_stud = self.client.get('/api/students')
        self.assertEqual(res_stud.status_code, 200)
        self.assertEqual(json.loads(res_stud.data)['students'], [])

        # Fetch empty attendance list
        res_att = self.client.get('/api/attendance')
        self.assertEqual(res_att.status_code, 200)
        self.assertEqual(json.loads(res_att.data)['attendance'], [])

if __name__ == '__main__':
    unittest.main()
