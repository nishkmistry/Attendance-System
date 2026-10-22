import unittest
from unittest.mock import patch, MagicMock
import io
import os
import json
import time
import shutil
import tempfile
import numpy as np
import cv2
import base64
import database
from face_recognition_module import FaceRecognitionSystem
from app import app, face_system, drone_stream, DroneStream

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

    def _textured_face(self, seed=0):
        """A crop that passes the brightness/sharpness quality gates (a flat gray one doesn't)."""
        rng = np.random.default_rng(seed)
        return rng.integers(60, 200, size=(120, 120, 3), dtype=np.uint8)

    def _det(self, box=(10, 10, 100, 100), seed=0):
        """One detect_faces_dnn result: native-res crop plus the aligned crop that gets embedded."""
        face = self._textured_face(seed)
        return {'box': box, 'face': face, 'aligned': cv2.resize(face, (112, 112)), 'confidence': 0.99}

    def _dummy_detection(self, box=(10, 10, 100, 100)):
        return [self._det(box)]

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
        mock_app_detect.return_value = self._dummy_detection(box=(10, 10, 100, 100))
        mock_mod_detect.return_value = self._dummy_detection(box=(10, 10, 100, 100))
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
            self._det((10, 10, 100, 100), 1),
            self._det((150, 10, 100, 100), 2),
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

    def _register(self, name, reg_no, embedding, mock_app_detect, mock_extract_embedding, seed=0):
        mock_app_detect.return_value = [self._det((10, 10, 100, 100), seed)]
        mock_extract_embedding.side_effect = None
        mock_extract_embedding.return_value = embedding
        return self.client.post('/api/register', json={
            'name': name, 'reg_no': reg_no, 'images': [self._create_sample_image_b64()] * 3
        })

    @patch('face_recognition_module.extract_embedding_from_crop')
    @patch('face_recognition_module.detect_faces_dnn')
    @patch('app.detect_faces_dnn')
    def test_lookalike_students_are_not_marked(self, mock_app_detect, mock_mod_detect, mock_extract_embedding):
        """A face nearly equidistant from two students is ambiguous - mark neither."""
        base = np.zeros(128, dtype=np.float32)
        a = base.copy(); a[0] = 1.0
        b = base.copy(); b[1] = 1.0
        self.assertEqual(self._register('Eve', 'R1', a, mock_app_detect, mock_extract_embedding, 1).status_code, 200)
        self.assertEqual(self._register('Finn', 'R2', b, mock_app_detect, mock_extract_embedding, 2).status_code, 200)

        between = a + b  # equally close to both
        mock_mod_detect.return_value = self._dummy_detection()
        mock_extract_embedding.return_value = between
        old_threshold = face_system.distance_threshold
        face_system.distance_threshold = 0.5  # put `between` (distance ~0.29) inside it
        try:
            att = json.loads(self.client.post('/api/attendance', json={'image': self._create_sample_image_b64()}).data)
        finally:
            face_system.distance_threshold = old_threshold
        self.assertEqual(att['marked_count'], 0)
        self.assertEqual(att['faces'][0]['status'], 'unknown')
        self.assertEqual(att['faces'][0]['reason'], 'ambiguous')

    @patch('face_recognition_module.extract_embedding_from_crop')
    @patch('face_recognition_module.detect_faces_dnn')
    @patch('app.detect_faces_dnn')
    def test_two_faces_cannot_both_be_one_student(self, mock_app_detect, mock_mod_detect, mock_extract_embedding):
        known = np.ones(128, dtype=np.float32)
        self.assertEqual(self._register('Gus', 'R3', known, mock_app_detect, mock_extract_embedding).status_code, 200)

        near = known.copy(); near[0] = 0.5
        mock_mod_detect.return_value = [
            self._det((10, 10, 100, 100), 1),
            self._det((150, 10, 100, 100), 2),
        ]
        mock_extract_embedding.side_effect = [near, known]
        att = json.loads(self.client.post('/api/attendance', json={'image': self._create_sample_image_b64()}).data)
        self.assertEqual(att['marked_count'], 1)
        self.assertEqual(sorted(f.get('reason') or 'matched' for f in att['faces']), ['duplicate', 'matched'])
        # The closer face (the exact embedding, second detection) keeps the match
        self.assertEqual(att['faces'][1]['status'], 'marked')

    @patch('face_recognition_module.extract_embedding_from_crop')
    @patch('app.detect_faces_dnn')
    def test_register_rejects_face_of_another_student(self, mock_app_detect, mock_extract_embedding):
        emb = np.ones(128, dtype=np.float32)
        self.assertEqual(self._register('Hana', 'R4', emb, mock_app_detect, mock_extract_embedding).status_code, 200)
        res = self._register('Ivan', 'R5', emb, mock_app_detect, mock_extract_embedding)
        self.assertEqual(res.status_code, 409)
        self.assertIn('already registered as Hana', json.loads(res.data)['message'])

    @patch('face_recognition_module.extract_embedding_from_crop')
    @patch('app.detect_faces_dnn')
    def test_reregister_replaces_face_samples(self, mock_app_detect, mock_extract_embedding):
        old = np.ones(128, dtype=np.float32)
        new = np.array([1.0] + [-1.0] * 127, dtype=np.float32)
        self.assertEqual(self._register('Jo', 'R6', old, mock_app_detect, mock_extract_embedding).status_code, 200)
        res = self._register('Jo', 'R6', new, mock_app_detect, mock_extract_embedding)
        self.assertEqual(res.status_code, 200)
        sid = json.loads(res.data)['student']['id']
        cached = face_system.embeddings_cache[sid]
        self.assertTrue(all(np.allclose(e, new) for e in cached))
        self.assertEqual(len(os.listdir(os.path.join(self.test_dataset, f'student_{sid}'))), len(cached))

        # Same reg no under a different name is still refused
        res2 = self._register('Someone Else', 'R6', new, mock_app_detect, mock_extract_embedding)
        self.assertEqual(res2.status_code, 409)

    @patch('app.detect_faces_dnn')
    def test_register_rejects_dark_face(self, mock_app_detect):
        dark = np.full((120, 120, 3), 12, dtype=np.uint8)
        dark[::2] += 8  # some texture, still far too dark
        mock_app_detect.return_value = [{'box': (10, 10, 100, 100), 'face': dark, 'confidence': 0.99}]
        res = self.client.post('/api/register', json={'name': 'Kai', 'reg_no': 'R7', 'image': self._create_sample_image_b64()})
        self.assertEqual(res.status_code, 400)
        self.assertIn('too dark', json.loads(res.data)['message'])

    @patch('app.detect_faces_dnn')
    def test_register_rejects_two_people_in_frame(self, mock_app_detect):
        mock_app_detect.return_value = [
            self._det((10, 10, 100, 100), 1),
            self._det((150, 10, 90, 90), 2),
        ]
        res = self.client.post('/api/register', json={'name': 'Lee', 'reg_no': 'R8', 'image': self._create_sample_image_b64()})
        self.assertEqual(res.status_code, 400)
        self.assertIn('More than one face', json.loads(res.data)['message'])

    def test_aligned_face_crop_levels_tilted_eyes(self):
        from face_recognition_module import aligned_face_crop
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Two "eyes" on a line tilted ~20 degrees
        cv2.circle(img, (120, 160), 6, (255, 255, 255), -1)   # right eye (image left)
        cv2.circle(img, (180, 182), 6, (255, 255, 255), -1)   # left eye (image right)
        crop = aligned_face_crop(img, (90, 110, 120, 120), left_eye=(180, 182), right_eye=(120, 160))
        self.assertEqual(crop.shape[:2], (120, 120))
        ys = [int(np.argmax(crop[:, x, 0])) for x in (30, 90) if crop[:, x, 0].max() > 0]
        self.assertTrue(len(ys) == 2 and abs(ys[0] - ys[1]) <= 3, ys)

    def test_rebuild_skips_unusable_stored_crops(self):
        good_dir = os.path.join(self.test_dataset, 'student_1')
        bad_dir = os.path.join(self.test_dataset, 'student_2')
        os.makedirs(good_dir); os.makedirs(bad_dir)
        cv2.imwrite(os.path.join(good_dir, '1.jpg'), cv2.resize(self._textured_face(), (112, 112)))
        cv2.imwrite(os.path.join(bad_dir, '1.jpg'), np.full((112, 112, 3), 15, dtype=np.uint8))
        with open(face_system.embeddings_cache_path, 'wb') as f:
            import pickle
            pickle.dump({1: [np.ones(128)]}, f)  # legacy untagged cache format
        face_system.load_model()
        with patch('face_recognition_module.extract_embedding_from_crop', return_value=np.ones(128)):
            missing = face_system.rebuild_embeddings()
        self.assertEqual(missing, [2])
        self.assertEqual(list(face_system.embeddings_cache), [1])

    def test_get_students_and_attendance_endpoints(self):
        # Fetch empty students list
        res_stud = self.client.get('/api/students')
        self.assertEqual(res_stud.status_code, 200)
        self.assertEqual(json.loads(res_stud.data)['students'], [])

        # Fetch empty attendance list
        res_att = self.client.get('/api/attendance')
        self.assertEqual(res_att.status_code, 200)
        self.assertEqual(json.loads(res_att.data)['attendance'], [])

    def _write_test_video(self, path, frames=10, fps=20):
        """Short clip whose frame i is filled with grey level 20*i, so the playback position is visible."""
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (64, 48))
        for i in range(frames):
            writer.write(np.full((48, 64, 3), 20 * i, dtype=np.uint8))
        writer.release()

    def test_drone_camera_endpoints(self):
        original = drone_stream.source
        try:
            # 1. Test drone config GET and POST
            res_get_cfg = self.client.get('/api/drone/config')
            self.assertEqual(res_get_cfg.status_code, 200)
            self.assertTrue(json.loads(res_get_cfg.data)['success'])

            # Quotes from Windows "Copy as path" are stripped
            res_post_cfg = self.client.post('/api/drone/config', json={'stream_url': '"rtsp://10.0.0.1:554/live"'})
            self.assertEqual(res_post_cfg.status_code, 200)
            data_cfg = json.loads(res_post_cfg.data)
            self.assertTrue(data_cfg['success'])
            self.assertEqual(data_cfg['stream_url'], 'rtsp://10.0.0.1:554/live')
            self.assertEqual(data_cfg['mode'], 'live')

            # 2. Test drone snapshot
            dummy_frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
            with patch.object(drone_stream, 'get_frame', return_value=dummy_frame):
                res_snap = self.client.get('/api/drone/snapshot')
            self.assertEqual(res_snap.status_code, 200)
            data_snap = json.loads(res_snap.data)
            self.assertTrue(data_snap['success'])
            self.assertTrue(data_snap['image'].startswith('data:image/jpeg;base64,'))

            with patch.object(drone_stream, 'get_frame', return_value=None):
                res_offline = self.client.get('/api/drone/snapshot')
            self.assertEqual(res_offline.status_code, 503)
        finally:
            drone_stream.set_source(original)

    def test_video_file_plays_in_real_time_and_loops(self):
        tmp = tempfile.mkdtemp()
        clip = os.path.join(tmp, 'clip.avi')
        self._write_test_video(clip)  # 10 frames at 20 fps = 0.5 s
        stream = DroneStream(clip)
        try:
            self.assertTrue(stream.is_video_file)
            self.assertIsNotNone(stream.get_frame(wait=3.0))
            levels = []
            end = time.monotonic() + 1.6
            while time.monotonic() < end:
                levels.append(round(stream.get_frame().mean() / 20))
                time.sleep(0.02)
        finally:
            reader = stream._thread
            stream.set_source('')
            if reader is not None:
                reader.join(3.0)
            shutil.rmtree(tmp, ignore_errors=True)

        # Plays through to the last frame, then starts over - about 3 times in 1.6 s.
        # Unpaced playback would jump around and "restart" on most samples.
        self.assertIn(9, levels)
        restarts = sum(1 for a, b in zip(levels, levels[1:]) if b < a)
        self.assertGreaterEqual(restarts, 2)
        self.assertLessEqual(restarts, 4)

    def test_upload_video_becomes_drone_source(self):
        tmp = tempfile.mkdtemp()
        upload_dir = os.path.join(tmp, 'uploads')
        clip = os.path.join(tmp, 'clip.avi')
        self._write_test_video(clip)
        original = drone_stream.source
        try:
            with patch('app.UPLOAD_DIR', upload_dir):
                with open(clip, 'rb') as f:
                    res = self.client.post('/api/drone/upload', data={'video': (f, 'Class Demo.avi')},
                                           content_type='multipart/form-data')
                data = res.get_json()
                self.assertEqual(res.status_code, 200, data)
                self.assertEqual(data['mode'], 'video')
                self.assertTrue(data['stream_url'].endswith('Class_Demo.avi'))
                self.assertEqual(drone_stream.source, data['stream_url'])

                res_missing = self.client.post('/api/drone/upload', data={}, content_type='multipart/form-data')
                self.assertEqual(res_missing.status_code, 400)

                res_wrong_type = self.client.post('/api/drone/upload', data={'video': (io.BytesIO(b'hello'), 'notes.txt')},
                                                  content_type='multipart/form-data')
                self.assertEqual(res_wrong_type.status_code, 400)

                # Right extension but not decodable: rejected and not left on disk
                res_corrupt = self.client.post('/api/drone/upload', data={'video': (io.BytesIO(b'not a video'), 'fake.mp4')},
                                               content_type='multipart/form-data')
                self.assertEqual(res_corrupt.status_code, 400)
                self.assertEqual(len(os.listdir(upload_dir)), 1)
                self.assertEqual(drone_stream.source, data['stream_url'])
        finally:
            drone_stream.set_source(original)
            shutil.rmtree(tmp, ignore_errors=True)

if __name__ == '__main__':
    unittest.main()
