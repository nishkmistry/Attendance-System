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

        conn = database.get_db_connection(self.test_db)
        conn.execute('DELETE FROM attendance')
        conn.execute('DELETE FROM students')
        conn.commit()
        conn.close()

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
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        _, buffer = cv2.imencode('.jpg', img)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"

    def test_database_operations(self):
        dummy_emb = np.random.randn(128).astype(np.float32)
        student_id = database.add_student('REG001', 'Alice', embedding=dummy_emb, db_path=self.test_db)
        self.assertIsNotNone(student_id)

        student = database.get_student_by_id(student_id, db_path=self.test_db)
        self.assertEqual(student['name'], 'Alice')
        self.assertEqual(student['reg_no'], 'REG001')

        candidates = database.get_candidate_embeddings(db_path=self.test_db)
        self.assertIn(student_id, candidates)
        self.assertEqual(len(candidates[student_id]), 128)

        mock_web3 = {
            'tx_hash': '0x123abc',
            'block_number': 12,
            'gas_used': 51000,
            'location_hash': '0xloc',
            'latency_ms': 15.0
        }

        rec1 = database.mark_attendance(student_id, web3_data=mock_web3, db_path=self.test_db)
        self.assertFalse(rec1['already_marked'])
        self.assertEqual(rec1['name'], 'Alice')
        self.assertEqual(rec1['tx_hash'], '0x123abc')

        rec2 = database.mark_attendance(student_id, web3_data=mock_web3, db_path=self.test_db)
        self.assertTrue(rec2['already_marked'])

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

    @patch('face_recognition_module.check_liveness')
    @patch('face_recognition_module.detect_faces')
    @patch('app.detect_faces')
    def test_register_and_mark_attendance_flow(self, mock_app_detect, mock_mod_detect, mock_liveness):
        mock_liveness.return_value = (True, 0.25, "Liveness verified")

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

        # 2. Mark attendance
        att_response = self.client.post('/api/attendance', json={
            'image': face_img_b64
        })
        self.assertEqual(att_response.status_code, 200)
        att_data = json.loads(att_response.data)
        self.assertTrue(att_data['success'])
        self.assertTrue(att_data['registered'])
        self.assertEqual(att_data['student']['name'], 'Charlie')
        self.assertIn('web3', att_data)
        self.assertTrue(att_data['web3']['success'])

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

    def test_web3_status_and_benchmarks_endpoints(self):
        res_w3 = self.client.get('/api/web3/status')
        self.assertEqual(res_w3.status_code, 200)
        data_w3 = json.loads(res_w3.data)
        self.assertTrue(data_w3['success'])
        self.assertIn('contract_address', data_w3['web3_status'])

        res_bm = self.client.get('/api/benchmarks')
        self.assertEqual(res_bm.status_code, 200)
        data_bm = json.loads(res_bm.data)
        self.assertTrue(data_bm['success'])
        self.assertIn('inference_latency', data_bm['benchmarks'])
        self.assertIn('gas_analysis', data_bm['benchmarks'])

if __name__ == '__main__':
    unittest.main()
