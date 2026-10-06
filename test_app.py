import unittest
from unittest.mock import patch, MagicMock
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
        face_system.load_model()

        app.config['TESTING'] = True
        self.client = app.test_client()

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
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

    def test_crop_circular_face(self):
        from face_recognition_module import crop_circular_face
        # Create test image with face-like region
        img = np.ones((200, 200, 3), dtype=np.uint8) * 150

        # Test crop with specific face box
        cropped, box = crop_circular_face(img, face_box=(20, 20, 60, 60), size=100, circular=True)
        self.assertIsNotNone(cropped)
        self.assertEqual(cropped.shape, (100, 100, 3))
        self.assertEqual(box, (20, 20, 60, 60))

    def _dummy_detection(self, box=(10, 10, 50, 50)):
        dummy_face_crop = np.ones((100, 100, 3), dtype=np.uint8) * 120
        return [{'box': box, 'face': dummy_face_crop, 'confidence': 0.99}]

    @patch('app.detect_faces_dnn')
    def test_detect_face_endpoint(self, mock_app_detect):
        sample_img_b64 = self._create_sample_image_b64()

        # Test case 1: Face detected
        mock_app_detect.return_value = self._dummy_detection()
        res = self.client.post('/api/detect_face', json={'image': sample_img_b64})
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertTrue(data['detected'])
        self.assertEqual(len(data['faces']), 1)
        self.assertIn('cropped_face', data)

        # Test case 2: No face detected
        mock_app_detect.return_value = []
        res2 = self.client.post('/api/detect_face', json={'image': sample_img_b64})
        self.assertEqual(res2.status_code, 200)
        data2 = json.loads(res2.data)
        self.assertTrue(data2['success'])
        self.assertFalse(data2['detected'])

    @patch('app.detect_faces_dnn')
    def test_register_student_no_face(self, mock_detect_faces):
        mock_detect_faces.return_value = []
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

    @patch('face_recognition_module.extract_embedding_from_crop')
    @patch('face_recognition_module.detect_faces_dnn')
    @patch('app.detect_faces_dnn')
    def test_register_and_mark_attendance_flow(self, mock_app_detect, mock_mod_detect, mock_extract_embedding):
        mock_detection = self._dummy_detection()
        mock_app_detect.return_value = mock_detection
        mock_mod_detect.return_value = mock_detection
        # Return identical embeddings for exact match
        dummy_embedding = np.ones((128,), dtype=np.float32)
        mock_extract_embedding.return_value = dummy_embedding

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

        # 2. Mark attendance using the same facial image (single face in frame)
        att_response = self.client.post('/api/attendance', json={
            'image': face_img_b64
        })
        self.assertEqual(att_response.status_code, 200)
        att_data = json.loads(att_response.data)
        self.assertTrue(att_data['success'])
        self.assertEqual(att_data['marked_count'], 1)
        self.assertEqual(len(att_data['faces']), 1)
        self.assertEqual(att_data['faces'][0]['status'], 'marked')
        self.assertEqual(att_data['faces'][0]['name'], 'Charlie')
        self.assertEqual(att_data['faces'][0]['reg_no'], 'REG003')

        # 3. Mark attendance again same day -> already_marked
        att_response2 = self.client.post('/api/attendance', json={'image': face_img_b64})
        att_data2 = json.loads(att_response2.data)
        self.assertEqual(att_data2['marked_count'], 0)
        self.assertEqual(att_data2['faces'][0]['status'], 'already_marked')

    @patch('face_recognition_module.extract_embedding_from_crop')
    @patch('face_recognition_module.detect_faces_dnn')
    def test_unregistered_student_attendance(self, mock_mod_detect, mock_extract_embedding):
        mock_mod_detect.return_value = self._dummy_detection()
        mock_extract_embedding.return_value = np.ones((128,), dtype=np.float32)

        sample_img_b64 = self._create_sample_image_b64()
        att_response = self.client.post('/api/attendance', json={
            'image': sample_img_b64
        })
        self.assertEqual(att_response.status_code, 200)
        att_data = json.loads(att_response.data)
        self.assertTrue(att_data['success'])
        self.assertEqual(att_data['unknown_count'], 1)
        self.assertEqual(att_data['faces'][0]['status'], 'unknown')
        self.assertEqual(att_data['faces'][0]['name'], 'Unknown1')

    @patch('face_recognition_module.extract_embedding_from_crop')
    @patch('face_recognition_module.detect_faces_dnn')
    @patch('app.detect_faces_dnn')
    def test_multi_face_batch_attendance(self, mock_app_detect, mock_mod_detect, mock_extract_embedding):
        """One frame with two faces: one registered (matches cached embedding), one unknown."""
        known_embedding = np.ones((128,), dtype=np.float32)
        unknown_embedding = np.array([1.0] + [-1.0] * 127, dtype=np.float32)  # far from known_embedding

        # Register a known student using face box A
        mock_app_detect.return_value = self._dummy_detection(box=(10, 10, 50, 50))
        mock_mod_detect.return_value = self._dummy_detection(box=(10, 10, 50, 50))
        mock_extract_embedding.return_value = known_embedding

        face_img_b64 = self._create_sample_image_b64()
        reg_response = self.client.post('/api/register', json={
            'name': 'Dana',
            'reg_no': 'REG004',
            'image': face_img_b64
        })
        self.assertTrue(json.loads(reg_response.data)['success'])

        # One frame containing both a known face and an unknown face
        two_faces = [
            {'box': (10, 10, 50, 50), 'face': np.ones((100, 100, 3), dtype=np.uint8) * 120, 'confidence': 0.99},
            {'box': (70, 70, 50, 50), 'face': np.ones((100, 100, 3), dtype=np.uint8) * 60, 'confidence': 0.99},
        ]
        mock_mod_detect.return_value = two_faces
        mock_extract_embedding.side_effect = [known_embedding, unknown_embedding]

        att_response = self.client.post('/api/attendance', json={'image': face_img_b64})
        att_data = json.loads(att_response.data)

        self.assertTrue(att_data['success'])
        self.assertEqual(att_data['marked_count'], 1)
        self.assertEqual(att_data['unknown_count'], 1)
        self.assertEqual(len(att_data['faces']), 2)
        statuses = sorted(f['status'] for f in att_data['faces'])
        self.assertEqual(statuses, ['marked', 'unknown'])

    def test_get_students_and_attendance_endpoints(self):
        # Fetch empty students list
        res_stud = self.client.get('/api/students')
        self.assertEqual(res_stud.status_code, 200)
        self.assertEqual(json.loads(res_stud.data)['students'], [])

        # Fetch empty attendance list
        res_att = self.client.get('/api/attendance')
        self.assertEqual(res_att.status_code, 200)
        self.assertEqual(json.loads(res_att.data)['attendance'], [])

    @patch('app.get_drone_cap')
    def test_drone_camera_endpoints(self, mock_get_drone_cap):
        # 1. Test drone config GET and POST
        res_get_cfg = self.client.get('/api/drone/config')
        self.assertEqual(res_get_cfg.status_code, 200)
        self.assertTrue(json.loads(res_get_cfg.data)['success'])

        res_post_cfg = self.client.post('/api/drone/config', json={'stream_url': 'rtsp://10.0.0.1:554/live'})
        self.assertEqual(res_post_cfg.status_code, 200)
        data_cfg = json.loads(res_post_cfg.data)
        self.assertTrue(data_cfg['success'])
        self.assertEqual(data_cfg['stream_url'], 'rtsp://10.0.0.1:554/live')

        # 2. Test drone snapshot
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
        mock_cap.read.return_value = (True, dummy_frame)
        mock_get_drone_cap.return_value = mock_cap

        res_snap = self.client.get('/api/drone/snapshot')
        self.assertEqual(res_snap.status_code, 200)
        data_snap = json.loads(res_snap.data)
        self.assertTrue(data_snap['success'])
        self.assertTrue(data_snap['image'].startswith('data:image/jpeg;base64,'))

if __name__ == '__main__':
    unittest.main()
