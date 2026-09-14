import cv2
import numpy as np
import os
import base64
import urllib.request

CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'
LOCAL_CASCADE_PATH = os.path.join(os.path.dirname(__file__), CASCADE_FILENAME)

def ensure_cascade_exists():
    """Ensure Haar Cascade xml file is available locally or download if needed."""
    if os.path.exists(LOCAL_CASCADE_PATH):
        return LOCAL_CASCADE_PATH

    cv2_data_path = getattr(cv2, 'data', None)
    if cv2_data_path and hasattr(cv2_data_path, 'haarcascades'):
        p = os.path.join(cv2_data_path.haarcascades, CASCADE_FILENAME)
        if os.path.exists(p):
            return p

    # Download if not present
    url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{CASCADE_FILENAME}"
    try:
        urllib.request.urlretrieve(url, LOCAL_CASCADE_PATH)
        return LOCAL_CASCADE_PATH
    except Exception as e:
        print(f"Warning: Could not download cascade xml: {e}")
        return LOCAL_CASCADE_PATH

cascade_path = ensure_cascade_exists()
face_cascade = cv2.CascadeClassifier(cascade_path)

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces, gray

class FaceRecognitionSystem:
    def __init__(self, dataset_dir='dataset', model_path='trainer.yml', distance_threshold=85.0):
        self.dataset_dir = dataset_dir
        self.model_path = model_path
        self.distance_threshold = distance_threshold
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.load_model()

    def load_model(self):
        """Loads trained LBPH model if file exists."""
        if os.path.exists(self.model_path):
            try:
                self.recognizer.read(self.model_path)
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False

    def save_student_face(self, student_id, gray_face):
        """
        Saves a cropped grayscale face image for a student and retrains model.
        """
        student_dir = os.path.join(self.dataset_dir, f"student_{student_id}")
        os.makedirs(student_dir, exist_ok=True)

        existing_files = [f for f in os.listdir(student_dir) if f.endswith('.jpg')]
        img_idx = len(existing_files) + 1
        img_path = os.path.join(student_dir, f"{img_idx}.jpg")

        resized_face = cv2.resize(gray_face, (200, 200))
        cv2.imwrite(img_path, resized_face)

        return self.train_model()

    def train_model(self):
        """
        Trains LBPH recognizer on all student face images in dataset_dir.
        """
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

    def recognize_face_from_image(self, b64_or_image):
        """
        Given a base64 image string or BGR OpenCV image, detects face and predicts student_id.
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

        faces, gray = detect_faces(image)
        if len(faces) == 0:
            return {
                'status': 'no_face',
                'student_id': None,
                'confidence': 0,
                'message': 'No face detected in the image'
            }

        if not os.path.exists(self.model_path):
            return {
                'status': 'not_registered',
                'student_id': None,
                'confidence': 0,
                'message': 'Student is not registered'
            }

        # Select largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        face_roi = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (200, 200))

        try:
            label_id, distance = self.recognizer.predict(resized_face)
        except Exception as e:
            return {
                'status': 'not_registered',
                'student_id': None,
                'confidence': 0,
                'message': f'Recognition error: {str(e)}'
            }

        if distance <= self.distance_threshold:
            return {
                'status': 'success',
                'student_id': label_id,
                'confidence': round(distance, 2),
                'message': 'Face recognized successfully'
            }
        else:
            return {
                'status': 'not_registered',
                'student_id': None,
                'confidence': round(distance, 2),
                'message': 'Student is not registered'
            }
