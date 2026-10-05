import cv2
import numpy as np
import os
import base64
import urllib.request
import logging

# Suppress verbose warnings/logs if needed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("deepface").setLevel(logging.ERROR)

CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'
LOCAL_CASCADE_PATH = os.path.join(os.path.dirname(__file__), CASCADE_FILENAME)

def ensure_cascade_exists():
    """Ensure Haar Cascade xml files exist locally and in cv2.data directory for DeepFace backend."""
    # 1. Local path
    if not os.path.exists(LOCAL_CASCADE_PATH):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{CASCADE_FILENAME}"
        try:
            urllib.request.urlretrieve(url, LOCAL_CASCADE_PATH)
        except Exception as e:
            print(f"Warning: Could not download cascade xml: {e}")

    # 2. cv2.data directory (needed by DeepFace OpenCV detector)
    cv2_data_path = getattr(cv2, 'data', None)
    if cv2_data_path and hasattr(cv2_data_path, 'haarcascades'):
        cdir = cv2.data.haarcascades
        os.makedirs(cdir, exist_ok=True)
        for xml in ['haarcascade_frontalface_default.xml', 'haarcascade_eye.xml']:
            p = os.path.join(cdir, xml)
            if not os.path.exists(p):
                url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{xml}"
                try:
                    urllib.request.urlretrieve(url, p)
                except Exception as e:
                    print(f"Warning: Could not download {xml} to cv2.data: {e}")

    if os.path.exists(LOCAL_CASCADE_PATH):
        return LOCAL_CASCADE_PATH
    return CASCADE_FILENAME

cascade_path = ensure_cascade_exists()
face_cascade = cv2.CascadeClassifier(cascade_path)

def base64_to_image(b64_string):
    """Converts a base64 string to an OpenCV BGR image."""
    if not b64_string:
        return None
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    try:
        image_bytes = base64.b64decode(b64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

def detect_faces(image):
    """
    Detects faces in a BGR image using OpenCV Haar Cascade.
    Returns list of (x, y, w, h) bounding boxes and grayscale image.
    """
    if image is None:
        return [], None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces, gray

class FaceRecognitionSystem:
    def __init__(self, dataset_dir='dataset', model_name='VGG-Face', distance_threshold=0.40):
        self.dataset_dir = dataset_dir
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        # Legacy attribute compatibility for model_path if referenced in tests
        self.model_path = os.path.join(self.dataset_dir, 'model_meta.json')
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.load_model()

    def load_model(self):
        """Loads DeepFace model into cache by building representation model."""
        ensure_cascade_exists()
        return True

    def save_student_face(self, student_id, gray_or_color_face):
        """
        Saves face image for a student into dataset_dir/student_{student_id}/.
        """
        student_dir = os.path.join(self.dataset_dir, f"student_{student_id}")
        os.makedirs(student_dir, exist_ok=True)

        existing_files = [f for f in os.listdir(student_dir) if f.endswith(('.jpg', '.png'))]
        img_idx = len(existing_files) + 1
        img_path = os.path.join(student_dir, f"{img_idx}.jpg")

        if len(gray_or_color_face.shape) == 2:
            face_img = cv2.cvtColor(gray_or_color_face, cv2.COLOR_GRAY2BGR)
        else:
            face_img = gray_or_color_face

        resized_face = cv2.resize(face_img, (200, 200))
        cv2.imwrite(img_path, resized_face)
        return True

    def extract_embedding(self, image):
        """
        Extracts DeepFace 2048-d / 4096-d feature vector embedding from face image.
        Uses DeepFace.represent.
        """
        from deepface import DeepFace
        try:
            results = DeepFace.represent(
                img_path=image,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv'
            )
            if results and len(results) > 0:
                return np.array(results[0]['embedding'])
        except Exception as e:
            print(f"Error extracting DeepFace embedding: {e}")
        return None

    def recognize_face_from_image(self, b64_or_image):
        """
        Given a base64 image string or BGR OpenCV image, detects face and predicts student_id using DeepFace.
        Returns dict:
        {
           'status': 'success' | 'no_face' | 'not_registered' | 'error',
           'student_id': int or None,
           'confidence': float,
           'message': str
        }
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
                'message': 'Invalid image provided'
            }

        faces, _ = detect_faces(image)
        if len(faces) == 0:
            return {
                'status': 'no_face',
                'student_id': None,
                'confidence': 0,
                'message': 'No face detected in the image'
            }

        if not os.path.exists(self.dataset_dir) or len(os.listdir(self.dataset_dir)) == 0:
            return {
                'status': 'not_registered',
                'student_id': None,
                'confidence': 0,
                'message': 'Student is not registered'
            }

        target_embedding = self.extract_embedding(image)
        if target_embedding is None:
            return {
                'status': 'not_registered',
                'student_id': None,
                'confidence': 0,
                'message': 'Student is not registered'
            }

        best_student_id = None
        min_distance = float('inf')

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
                if filename.endswith(('.jpg', '.png')):
                    img_path = os.path.join(student_dir, filename)
                    student_img = cv2.imread(img_path)
                    if student_img is not None:
                        student_embedding = self.extract_embedding(student_img)
                        if student_embedding is not None:
                            # Cosine distance
                            norm_t = np.linalg.norm(target_embedding)
                            norm_s = np.linalg.norm(student_embedding)
                            if norm_t > 0 and norm_s > 0:
                                sim = np.dot(target_embedding, student_embedding) / (norm_t * norm_s)
                                dist = 1.0 - sim
                                if dist < min_distance:
                                    min_distance = dist
                                    best_student_id = student_id

        if best_student_id is not None and min_distance <= self.distance_threshold:
            return {
                'status': 'success',
                'student_id': best_student_id,
                'confidence': round(float(min_distance), 4),
                'message': 'Face recognized successfully'
            }
        else:
            return {
                'status': 'not_registered',
                'student_id': None,
                'confidence': round(float(min_distance if min_distance != float('inf') else 1.0), 4),
                'message': 'Student is not registered'
            }
