import unittest
import numpy as np
import cv2
import face_recognition_module as frm

class BiometricsPipelineTestCase(unittest.TestCase):

    def test_ear_calculation(self):
        # Sample eye landmarks
        landmarks = [
            [10, 20],  # p1
            [20, 15],  # p2
            [30, 15],  # p3
            [40, 20],  # p4
            [30, 25],  # p5
            [20, 25]   # p6
        ]
        ear = frm.calculate_ear(landmarks)
        # ||p2 - p6|| = 10, ||p3 - p5|| = 10, ||p1 - p4|| = 30
        # EAR = (10 + 10) / (2 * 30) = 20 / 60 = 0.3333...
        self.assertAlmostEqual(ear, 1.0 / 3.0, places=3)

    def test_check_liveness(self):
        # Create dummy facial ROI
        dummy_roi = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(dummy_roi, (20, 20), (80, 80), (255, 255, 255), -1)

        is_live, ear_val, msg = frm.check_liveness(dummy_roi)
        self.assertIsInstance(is_live, bool)
        self.assertIsInstance(ear_val, float)
        self.assertTrue(len(msg) > 0)

    def test_128d_embedding_extraction(self):
        dummy_face = np.ones((160, 160, 3), dtype=np.uint8) * 150
        emb = frm.embedder.extract_embedding(dummy_face)

        self.assertEqual(len(emb), 128)
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_cosine_similarity(self):
        v1 = np.array([1.0, 0.0, 0.0] + [0.0]*125, dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0] + [0.0]*125, dtype=np.float32)
        v3 = np.array([0.0, 1.0, 0.0] + [0.0]*125, dtype=np.float32)

        sim_identical = frm.cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim_identical, 1.0, places=4)

        sim_orthogonal = frm.cosine_similarity(v1, v3)
        self.assertAlmostEqual(sim_orthogonal, 0.0, places=4)

    def test_vector_matching_engine_threshold(self):
        v_target = np.random.randn(128).astype(np.float32)
        v_target /= np.linalg.norm(v_target)

        # High similarity candidate (> 0.85)
        v_high = v_target + np.random.randn(128) * 0.05
        v_high /= np.linalg.norm(v_high)

        # Low similarity candidate (< 0.85)
        v_low = -v_target

        candidates = {
            1: v_high,
            2: v_low
        }

        matched_sid, score, is_match = frm.match_vector_embedding(v_target, candidates, similarity_threshold=0.85)
        self.assertTrue(is_match)
        self.assertEqual(matched_sid, 1)
        self.assertGreaterEqual(score, 0.85)

if __name__ == '__main__':
    unittest.main()
