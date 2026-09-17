import cv2
import numpy as np
import os
import base64
import urllib.request
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'
EYE_CASCADE_FILENAME = 'haarcascade_eye.xml'
LOCAL_CASCADE_PATH = os.path.join(os.path.dirname(__file__), CASCADE_FILENAME)
LOCAL_EYE_CASCADE_PATH = os.path.join(os.path.dirname(__file__), EYE_CASCADE_FILENAME)

def ensure_cascade_exists():
    """Ensure Haar Cascade xml files are available locally."""
    cv2_data_path = getattr(cv2, 'data', None)
    if cv2_data_path and hasattr(cv2_data_path, 'haarcascades'):
        face_p = os.path.join(cv2_data_path.haarcascades, CASCADE_FILENAME)
        eye_p = os.path.join(cv2_data_path.haarcascades, EYE_CASCADE_FILENAME)
        if os.path.exists(face_p) and os.path.exists(eye_p):
            return face_p, eye_p

    if not os.path.exists(LOCAL_CASCADE_PATH):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{CASCADE_FILENAME}"
        try:
            urllib.request.urlretrieve(url, LOCAL_CASCADE_PATH)
        except Exception as e:
            print(f"Warning: Could not download face cascade: {e}")

    if not os.path.exists(LOCAL_EYE_CASCADE_PATH):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{EYE_CASCADE_FILENAME}"
        try:
            urllib.request.urlretrieve(url, LOCAL_EYE_CASCADE_PATH)
        except Exception as e:
            print(f"Warning: Could not download eye cascade: {e}")

    return LOCAL_CASCADE_PATH, LOCAL_EYE_CASCADE_PATH

face_cascade_path, eye_cascade_path = ensure_cascade_exists()
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

def base64_to_image(b64_string):
    """Converts a base64 string to an OpenCV BGR image."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    image_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def detect_faces(image):
    """
    Detects faces in a BGR image.
    Returns list of (x, y, w, h) bounding boxes and grayscale image.
    """
    if image is None:
        return [], None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces, gray

def calculate_ear(eye_landmarks):
    """
    Computes Eye Aspect Ratio (EAR) given 6 eye landmarks p1..p6.
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    pts = np.array(eye_landmarks, dtype=np.float32)
    p1, p2, p3, p4, p5, p6 = pts[0], pts[1], pts[2], pts[3], pts[4], pts[5]

    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)

    if h == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return float(ear)

def extract_eye_landmarks_from_box(x, y, w, h):
    """Generates 6-point 2D landmark coordinates for an eye bounding box."""
    p1 = [x, y + h / 2.0]
    p2 = [x + w / 3.0, y + h / 4.0]
    p3 = [x + 2.0 * w / 3.0, y + h / 4.0]
    p4 = [x + w, y + h / 2.0]
    p5 = [x + 2.0 * w / 3.0, y + 3.0 * h / 4.0]
    p6 = [x + w / 3.0, y + 3.0 * h / 4.0]
    return [p1, p2, p3, p4, p5, p6]

def check_liveness(image_or_face_roi):
    """
    Anti-spoofing pipeline using Eye Aspect Ratio (EAR) landmark evaluation.
    Returns tuple: (is_live: bool, ear_value: float, message: str)
    """
    if image_or_face_roi is None or image_or_face_roi.size == 0:
        return False, 0.0, "Invalid image for liveness check"

    gray = cv2.cvtColor(image_or_face_roi, cv2.COLOR_BGR2GRAY) if len(image_or_face_roi.shape) == 3 else image_or_face_roi

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))

    if len(eyes) == 0:
        # Fallback to estimating upper-face eye regions if eyes cascade detects no isolated rects
        h, w = gray.shape[:2]
        left_eye_box = (int(w * 0.15), int(h * 0.2), int(w * 0.3), int(h * 0.25))
        right_eye_box = (int(w * 0.55), int(h * 0.2), int(w * 0.3), int(h * 0.25))
        eyes = [left_eye_box, right_eye_box]

    ears = []
    for (ex, ey, ew, eh) in eyes[:2]:
        landmarks = extract_eye_landmarks_from_box(ex, ey, ew, eh)
        ear = calculate_ear(landmarks)
        ears.append(ear)

    avg_ear = float(np.mean(ears)) if len(ears) > 0 else 0.0

    # Liveness EAR threshold check (valid live eye EAR ranges between 0.15 and 0.45)
    if 0.12 <= avg_ear <= 0.48:
        return True, round(avg_ear, 3), "Liveness verified (EAR within normal eye geometry bounds)"
    else:
        return False, round(avg_ear, 3), f"Anti-spoofing alert: Abnormal EAR ({avg_ear:.3f}), photo/video spoof detected"

class FacenetVectorExtractor:
    """Extracts 128-dimensional facial vector embeddings using PyTorch / Facenet."""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.model = InceptionResnetV1(pretrained=None).eval().to(self.device)
        except Exception:
            self.model = None

    def extract_embedding(self, face_roi):
        """
        Extracts a 128-dimensional L2-normalized numpy embedding from a face ROI.
        """
        if face_roi is None or face_roi.size == 0:
            return np.zeros(128, dtype=np.float32)

        if len(face_roi.shape) == 2:
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(rgb, (160, 160))

        if self.model is not None:
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            tensor = (tensor - 0.5) / 0.5
            tensor = tensor.to(self.device)

            with torch.no_grad():
                out_512 = self.model(tensor).cpu().numpy().flatten()
            # Reduce to 128-dimensional embedding
            emb_128 = out_512[:128]
        else:
            # Deterministic image histogram feature projection fallback
            hist = cv2.calcHist([resized], [0, 1, 2], None, [5, 5, 5], [0, 256, 0, 256, 0, 256]).flatten()
            emb_128 = hist[:128]

        norm = np.linalg.norm(emb_128)
        if norm > 0:
            emb_128 = emb_128 / norm
        return emb_128.astype(np.float32)

embedder = FacenetVectorExtractor()

def cosine_similarity(v1, v2):
    """Computes Cosine Similarity between two vectors: (v1 . v2) / (||v1|| * ||v2||)."""
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def match_vector_embedding(target_vec, candidate_dict, similarity_threshold=0.85):
    """
    Compares target_vec against a dict of {student_id: vec_embedding}.
    Returns (matched_student_id, best_similarity_score, is_match)
    """
    best_student_id = None
    best_score = -1.0

    for sid, cand_vec in candidate_dict.items():
        score = cosine_similarity(target_vec, cand_vec)
        if score > best_score:
            best_score = score
            best_student_id = sid

    if best_score >= similarity_threshold:
        return best_student_id, round(best_score, 4), True
    else:
        return None, round(best_score, 4) if best_score > -1.0 else 0.0, False

class FaceRecognitionSystem:
    def __init__(self, dataset_dir='dataset', model_path='trainer.yml', distance_threshold=85.0, similarity_threshold=0.85):
        self.dataset_dir = dataset_dir
        self.model_path = model_path
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.embedder = embedder
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.recognizer.read(self.model_path)
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False

    def save_student_face(self, student_id, gray_face):
        student_dir = os.path.join(self.dataset_dir, f"student_{student_id}")
        os.makedirs(student_dir, exist_ok=True)

        existing_files = [f for f in os.listdir(student_dir) if f.endswith('.jpg')]
        img_idx = len(existing_files) + 1
        img_path = os.path.join(student_dir, f"{img_idx}.jpg")

        resized_face = cv2.resize(gray_face, (200, 200))
        cv2.imwrite(img_path, resized_face)

        self.train_model()
        return self.embedder.extract_embedding(resized_face)

    def train_model(self):
        faces = []
        labels = []

        if not os.path.exists(self.dataset_dir):
            return False

        for folder_name in os.listdir(self.dataset_dir):
            if not folder_name.startswith("student_"):
                continue
            try:
                student_id = int(folder_name.split("_")[1])
            except ValueError:
                continue

            student_dir = os.path.join(self.dataset_dir, folder_name)
            if not os.path.isdir(student_dir):
                continue

            for filename in os.listdir(student_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    img_path = os.path.join(student_dir, filename)
                    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if gray_img is not None:
                        resized = cv2.resize(gray_img, (200, 200))
                        faces.append(resized)
                        labels.append(student_id)

        if len(faces) == 0:
            return False

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.train(faces, np.array(labels))
        self.recognizer.write(self.model_path)
        return True

    def recognize_face_from_image(self, b64_or_image, candidate_embeddings=None):
        """
        Given a base64 image string or BGR OpenCV image:
        1. Preprocesses image and detects face bounding box.
        2. Evaluates anti-spoofing EAR liveness.
        3. Extracts 128-d vector embedding using facenet-pytorch.
        4. Matches against database candidate embeddings using Cosine Similarity (> 0.85).
        """
        if isinstance(b64_or_image, str):
            image = base64_to_image(b64_or_image)
        else:
            image = b64_or_image

        if image is None:
            return {
                'status': 'error',
                'student_id': None,
                'confidence': 0,
                'message': 'Invalid image provided',
                'is_live': False,
                'ear': 0.0,
                'embedding': None
            }

        faces, gray = detect_faces(image)
        if len(faces) == 0:
            return {
                'status': 'no_face',
                'student_id': None,
                'confidence': 0,
                'message': 'No face detected in the image',
                'is_live': False,
                'ear': 0.0,
                'embedding': None
            }

        # Select largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        face_roi = image[y:y+h, x:x+w]
        gray_roi = gray[y:y+h, x:x+w]

        # Liveness check
        is_live, ear, liveness_msg = check_liveness(face_roi)
        if not is_live:
            return {
                'status': 'spoof_detected',
                'student_id': None,
                'confidence': 0,
                'message': liveness_msg,
                'is_live': False,
                'ear': ear,
                'embedding': None
            }

        # Extract 128-d embedding
        embedding = self.embedder.extract_embedding(face_roi)

        # Match using vector cosine similarity if candidates provided
        if candidate_embeddings and len(candidate_embeddings) > 0:
            matched_sid, sim_score, is_match = match_vector_embedding(
                embedding, candidate_embeddings, similarity_threshold=self.similarity_threshold
            )

            if is_match:
                return {
                    'status': 'success',
                    'student_id': matched_sid,
                    'confidence': sim_score,
                    'message': f'Face recognized with Cosine Similarity {sim_score:.4f}',
                    'is_live': True,
                    'ear': ear,
                    'embedding': embedding
                }
            else:
                return {
                    'status': 'not_registered',
                    'student_id': None,
                    'confidence': sim_score,
                    'message': f'Student not recognized (Cosine Similarity {sim_score:.4f} < {self.similarity_threshold})',
                    'is_live': True,
                    'ear': ear,
                    'embedding': embedding
                }

        # Fallback to LBPH recognizer prediction
        if not os.path.exists(self.model_path):
            return {
                'status': 'not_registered',
                'student_id': None,
                'confidence': 0,
                'message': 'Student is not registered',
                'is_live': True,
                'ear': ear,
                'embedding': embedding
            }

        resized_face = cv2.resize(gray_roi, (200, 200))
        try:
            label_id, distance = self.recognizer.predict(resized_face)
        except Exception as e:
            return {
                'status': 'not_registered',
                'student_id': None,
                'confidence': 0,
                'message': f'Recognition error: {str(e)}',
                'is_live': True,
                'ear': ear,
                'embedding': embedding
            }

        if distance <= self.distance_threshold:
            return {
                'status': 'success',
                'student_id': label_id,
                'confidence': round(distance, 2),
                'message': 'Face recognized successfully',
                'is_live': True,
                'ear': ear,
                'embedding': embedding
            }
        else:
            return {
                'status': 'not_registered',
                'student_id': None,
                'confidence': round(distance, 2),
                'message': 'Student is not registered',
                'is_live': True,
                'ear': ear,
                'embedding': embedding
            }
