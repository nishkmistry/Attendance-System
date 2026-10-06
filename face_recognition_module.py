import cv2
import numpy as np
import os
import sys
import base64
import pickle
import logging

# DeepFace prints emoji warnings (e.g. a tf-keras version notice) at import time.
# On a Windows console using the cp1252 codepage, printing that emoji raises
# UnicodeEncodeError and takes the whole `from deepface import DeepFace` down with it.
# Force UTF-8 stdio before DeepFace is ever imported, anywhere in this module.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

# Suppress verbose warnings/logs if needed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("deepface").setLevel(logging.ERROR)


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


def image_to_base64(image):
    """Converts an OpenCV BGR image to base64 Data URL string."""
    if image is None:
        return None
    ret, buffer = cv2.imencode('.jpg', image)
    if not ret:
        return None
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"


def detect_faces_dnn(image, max_dimension=800):
    """
    Detects ALL faces in a BGR image using DeepFace's MTCNN detector (weights bundled
    with the package, no external download - RetinaFace needs a ~100MB GitHub release
    download on first use, which is a single point of failure on a restricted network).
    Suited for multi-face, off-angle, small-face scenes (e.g. a drone shot of a classroom).

    The frame is downscaled before detection to keep CPU/memory use bounded regardless
    of the source resolution (a full-res drone frame run through this repeatedly, e.g.
    the live polling overlay, was observed to OOM on a memory-constrained machine).
    Returned boxes are scaled back up to the original image's coordinates.

    Returns a list of dicts: {'box': (x, y, w, h), 'face': aligned_face_bgr, 'confidence': float}
    """
    if image is None:
        return []

    img_h, img_w = image.shape[:2]
    scale = 1.0
    detect_img = image
    if max(img_h, img_w) > max_dimension:
        scale = max_dimension / float(max(img_h, img_w))
        detect_img = cv2.resize(image, (int(img_w * scale), int(img_h * scale)))

    from deepface import DeepFace
    try:
        results = DeepFace.extract_faces(
            img_path=detect_img,
            detector_backend='mtcnn',
            enforce_detection=False,
            align=True
        )
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return []

    faces = []
    for r in results or []:
        # DeepFace inserts a synthetic full-frame "face" with confidence=0 when
        # enforce_detection=False and nothing was actually detected - skip it,
        # otherwise every face-less frame falsely yields one Unknown entry.
        if float(r.get('confidence', 0) or 0) <= 0:
            continue

        area = r.get('facial_area', {}) or {}
        x = int(area.get('x', 0) / scale)
        y = int(area.get('y', 0) / scale)
        w = int(area.get('w', 0) / scale)
        h = int(area.get('h', 0) / scale)
        if w <= 0 or h <= 0:
            continue

        face_arr = r.get('face')
        if face_arr is None:
            continue
        face_arr = np.asarray(face_arr)
        if face_arr.dtype != np.uint8:
            if face_arr.size > 0 and face_arr.max() <= 1.0:
                face_arr = (face_arr * 255).astype(np.uint8)
            else:
                face_arr = face_arr.astype(np.uint8)
        face_bgr = cv2.cvtColor(face_arr, cv2.COLOR_RGB2BGR)

        faces.append({
            'box': (x, y, w, h),
            'face': face_bgr,
            'confidence': float(r.get('confidence', 1.0))
        })

    return faces


def extract_embedding_from_crop(face_bgr, model_name='Facenet'):
    """
    Extracts a face embedding from an already-cropped/aligned BGR face image.
    Uses detector_backend='skip' since the face is already located - avoids re-detecting
    (and re-aligning) a crop that was already produced by detect_faces_dnn, keeping
    registration-time and recognition-time embeddings consistently framed.
    """
    if face_bgr is None or face_bgr.size == 0:
        return None

    from deepface import DeepFace
    try:
        results = DeepFace.represent(
            img_path=face_bgr,
            model_name=model_name,
            enforce_detection=False,
            detector_backend='skip'
        )
        if results and len(results) > 0:
            return np.array(results[0]['embedding'])
    except Exception as e:
        print(f"Error extracting embedding from crop: {e}")
    return None


def crop_circular_face(image, face_box=None, size=200, padding=0.25, circular=True):
    """
    Crops a face from an image within a circular boundary, for UI preview thumbnails only.
    This output must never be used as the source for a recognition embedding - the circular
    mask's black corners distort the embedding and will cause recognition to fail.
    Returns (cropped_face_bgr, face_box).
    """
    if image is None:
        return None, None

    if face_box is None:
        detections = detect_faces_dnn(image)
        if len(detections) == 0:
            return None, None
        face_box = max(detections, key=lambda d: d['box'][2] * d['box'][3])['box']

    x, y, w, h = [int(v) for v in face_box]
    img_h, img_w = image.shape[:2]

    # Calculate bounding square with padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    face_crop = image[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, (x, y, w, h)

    resized_face = cv2.resize(face_crop, (size, size))

    if circular:
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (size // 2, size // 2), size // 2, 255, -1)
        circular_face = cv2.bitwise_and(resized_face, resized_face, mask=mask)
        return circular_face, (x, y, w, h)

    return resized_face, (x, y, w, h)


# DeepFace's own tuned cosine-distance thresholds per model (deepface/config/threshold.py),
# mirrored here so picking a threshold doesn't force-import deepface (and the TensorFlow
# chain behind it) just to construct a FaceRecognitionSystem - that import alone takes
# 20-40s on CPU, which must not happen just from `import app` (e.g. in tests).
_COSINE_THRESHOLDS = {
    'VGG-Face': 0.68, 'Facenet': 0.40, 'Facenet512': 0.30, 'ArcFace': 0.68,
    'Dlib': 0.07, 'SFace': 0.593, 'OpenFace': 0.10, 'DeepFace': 0.23,
    'DeepID': 0.015, 'GhostFaceNet': 0.65, 'Buffalo_L': 0.55,
}


class FaceRecognitionSystem:
    def __init__(self, dataset_dir='dataset', model_name='Facenet', distance_threshold=None):
        # Facenet over VGG-Face: VGG-Face's 4096-d FC layer was observed to OOM on a
        # memory-constrained machine; Facenet is far lighter (128-d) and just as accurate
        # for this use case. Threshold is looked up per model since different models have
        # very different cosine-distance distributions (VGG-Face's tuned threshold is
        # 0.68, not the 0.40 this app used to hardcode regardless of model).
        self.dataset_dir = dataset_dir
        self.model_name = model_name
        self.distance_threshold = (
            distance_threshold if distance_threshold is not None
            else _COSINE_THRESHOLDS.get(model_name, 0.40)
        )
        self.embeddings_cache = {}
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.load_model()

    @property
    def embeddings_cache_path(self):
        return os.path.join(self.dataset_dir, 'embeddings_cache.pkl')

    def load_model(self):
        """Loads cached student embeddings from disk into memory."""
        if os.path.exists(self.embeddings_cache_path):
            try:
                with open(self.embeddings_cache_path, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
            except Exception as e:
                print(f"Warning: could not load embeddings cache: {e}")
                self.embeddings_cache = {}
        return True

    def _persist_cache(self):
        with open(self.embeddings_cache_path, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)

    def warmup(self):
        """
        Forces TensorFlow/MTCNN/the embedding model to load now, at server startup,
        instead of lazily on the first real request - the cold start (model build +
        first-call graph tracing) can take 20-40s on CPU, which otherwise makes the
        very first registration or attendance scan look hung.
        """
        dummy = np.ones((300, 300, 3), dtype=np.uint8) * 127
        detect_faces_dnn(dummy)
        extract_embedding_from_crop(cv2.resize(dummy, (200, 200)), self.model_name)
        return True

    def save_student_face(self, student_id, face_crop_bgr):
        """
        Saves a student's rectangular (non-circular-masked) face crop to
        dataset_dir/student_{student_id}/ and caches its embedding in memory + on disk,
        so recognition never has to recompute it from the stored image again.
        """
        student_dir = os.path.join(self.dataset_dir, f"student_{student_id}")
        os.makedirs(student_dir, exist_ok=True)

        existing_files = [f for f in os.listdir(student_dir) if f.endswith(('.jpg', '.png'))]
        img_idx = len(existing_files) + 1
        img_path = os.path.join(student_dir, f"{img_idx}.jpg")

        if len(face_crop_bgr.shape) == 2:
            face_crop_bgr = cv2.cvtColor(face_crop_bgr, cv2.COLOR_GRAY2BGR)

        resized_face = cv2.resize(face_crop_bgr, (200, 200))
        cv2.imwrite(img_path, resized_face)

        embedding = extract_embedding_from_crop(resized_face, self.model_name)
        if embedding is not None:
            self.embeddings_cache.setdefault(student_id, []).append(embedding)
            self._persist_cache()

        return True

    def recognize_faces_batch(self, b64_or_image):
        """
        Detects every face in the given frame and matches each one against the cached
        student embeddings. Pure recognition - does not write attendance records.

        Returns a list of dicts: {'box': (x, y, w, h), 'student_id': int|None, 'distance': float}
        """
        if isinstance(b64_or_image, str):
            image = base64_to_image(b64_or_image)
        else:
            image = b64_or_image

        if image is None:
            return []

        detections = detect_faces_dnn(image)
        results = []

        for det in detections:
            embedding = extract_embedding_from_crop(det['face'], self.model_name)
            best_student_id = None
            min_distance = float('inf')

            if embedding is not None:
                norm_t = np.linalg.norm(embedding)
                if norm_t > 0:
                    for student_id, emb_list in self.embeddings_cache.items():
                        for stored_emb in emb_list:
                            norm_s = np.linalg.norm(stored_emb)
                            if norm_s == 0:
                                continue
                            sim = np.dot(embedding, stored_emb) / (norm_t * norm_s)
                            dist = 1.0 - sim
                            if dist < min_distance:
                                min_distance = dist
                                best_student_id = student_id

            matched = best_student_id is not None and min_distance <= self.distance_threshold
            results.append({
                'box': det['box'],
                'student_id': best_student_id if matched else None,
                'distance': round(float(min_distance if min_distance != float('inf') else 1.0), 4)
            })

        return results
