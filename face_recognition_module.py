import cv2
import numpy as np
import os
import re
import sys
import base64
import pickle
import hashlib
import logging
import threading
import urllib.request

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

# Face detection (YuNet) and the default embedding model (SFace) both run on OpenCV's DNN
# module - no TensorFlow. Measured on this deployment's phone-video footage of three
# enrolled students against the previous MTCNN + Facenet (DeepFace) pipeline: detection
# 16ms vs 544ms per frame, embedding 6ms vs 403ms per face; of 137 labelled faces of
# enrolled students, 134 matched the right student and none a wrong one (Facenet: 36
# right, 20 marked as somebody else), and no stranger in stock classroom footage came
# within reach of the threshold (best cosine similarity 0.24 vs 0.363 needed).
# SFace is trained on faces warped onto a fixed 5-landmark template (eyes, nose tip, mouth
# corners), which YuNet's landmarks supply.
# Overridable via FACE_MODEL with any DeepFace model name (e.g. Facenet, ArcFace); changing
# it triggers a one-off rebuild of the embeddings cache from the stored face crops (see
# FaceRecognitionSystem.load_model).
DEFAULT_MODEL = os.environ.get('FACE_MODEL', 'SFace')
# CLAHE on the face before embedding. Applies to DeepFace models only: it helped Facenet on
# a backlit webcam, but for SFace it let strangers land closer to enrolled students
# (best stranger similarity 0.32 with it vs 0.24 without) for no gain on genuine matches.
USE_LIGHTING_NORMALIZATION = True

# Side of the aligned square crop SFace takes (and that enrolment samples are stored as).
ALIGNED_SIZE = 112

# YuNet confidence below which a detection is dropped. Real faces in the footage scored
# 0.78+ (small classroom faces) and 0.84+ (phone video); hands, shoulders and sideways
# heads scored 0.60-0.84, so some of those still get through - as Unknown faces.
DETECTION_SCORE = 0.75

# Embeddings are only comparable if they were computed from identically prepared crops:
# bump this when that changes, so the cache gets rebuilt from the stored samples.
EMBEDDING_PIPELINE = 'yunet-5pt-v1'

# ONNX weights: file name, sha256, then download URLs tried in order. They live in
# DeepFace's weights folder, under the names DeepFace itself uses for these models. The
# Hugging Face copy is OpenCV's own mirror - GitHub serves these files from
# media.githubusercontent.com, which some networks (including this deployment's) block.
_WEIGHTS = {
    'yunet': ('face_detection_yunet_2023mar.onnx',
              '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4', [
                  'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
                  'https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx',
              ]),
    'sface': ('face_recognition_sface_2021dec.onnx',
              '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79', [
                  'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
                  'https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx',
              ]),
}

# OpenCV DNN nets must not run from several request threads at once; every use of the
# detector/recognizer below holds this lock (each call takes milliseconds).
_MODEL_LOCK = threading.Lock()
_models = {}


def _weights_file(key):
    """Local path of an ONNX model file, downloading (and checksum-verifying) it on first use."""
    fname, sha256, urls = _WEIGHTS[key]
    folder = os.path.join(os.environ.get('DEEPFACE_HOME', os.path.expanduser('~')), '.deepface', 'weights')
    path = os.path.join(folder, fname)
    if os.path.isfile(path):
        return path

    os.makedirs(folder, exist_ok=True)
    tmp = path + '.part'
    for url in urls:
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as res, open(tmp, 'wb') as out:
                for chunk in iter(lambda: res.read(1 << 20), b''):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != sha256:
                raise ValueError('checksum mismatch')
            os.replace(tmp, path)
            return path
        except Exception as e:
            print(f"Could not download {fname} from {url}: {e}")
    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError(f"Face model {fname} is missing. Download it from {urls[-1]} into {folder}")


def _detector():
    if 'yunet' not in _models:
        _models['yunet'] = cv2.FaceDetectorYN.create(_weights_file('yunet'), '', (320, 320), DETECTION_SCORE)
    return _models['yunet']


def _recognizer():
    if 'sface' not in _models:
        _models['sface'] = cv2.FaceRecognizerSF.create(_weights_file('sface'), '')
    return _models['sface']


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


def detect_faces_dnn(image, max_dimension=1280):
    """
    Detects ALL faces in a BGR image with OpenCV's YuNet detector, which also locates five
    landmarks per face. Suited for multi-face, small-face scenes (e.g. a drone shot of a
    classroom) and fast enough (~15ms per frame on CPU) to run on every scan.

    Frames larger than max_dimension are downscaled for detection only; boxes and
    landmarks are mapped back, and both crops are cut from the full-resolution frame,
    since a blurrier face gives a less distinctive embedding.

    Returns a list of dicts:
    {'box': (x, y, w, h), 'face': face_bgr, 'aligned': aligned_bgr, 'confidence': float}
    where 'face' is the box at native resolution with the eyes levelled (for quality checks
    and previews) and 'aligned' is the ALIGNED_SIZE-square landmark-aligned crop the
    embedding model takes.
    """
    if image is None:
        return []

    img_h, img_w = image.shape[:2]
    scale = min(1.0, max_dimension / float(max(img_h, img_w)))
    detect_img = cv2.resize(image, (int(img_w * scale), int(img_h * scale))) if scale < 1 else image

    faces = []
    with _MODEL_LOCK:
        detector = _detector()
        detector.setInputSize((detect_img.shape[1], detect_img.shape[0]))
        _, rows = detector.detect(detect_img)
        if rows is None:
            return []
        rows = rows.copy()
        rows[:, :14] /= scale  # box + landmarks back to full-resolution coordinates

        for row in rows:
            # YuNet boxes may overhang the frame edge; clip so the box and crop agree
            x1, y1 = max(0, int(row[0])), max(0, int(row[1]))
            x2, y2 = min(img_w, int(row[0] + row[2])), min(img_h, int(row[1] + row[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            box = (x1, y1, x2 - x1, y2 - y1)
            # Landmarks: right eye, left eye (the person's), nose tip, mouth corners
            face_bgr = aligned_face_crop(image, box, left_eye=(row[6], row[7]), right_eye=(row[4], row[5]))
            if face_bgr is None:
                continue
            faces.append({
                'box': box,
                'face': face_bgr,
                'aligned': _recognizer().alignCrop(image, row),
                'confidence': float(row[14]),
            })

    return faces


def aligned_face_crop(image, box, left_eye=None, right_eye=None):
    """
    Crops `box` out of `image`, rotated so the eyes are level - the same alignment
    DeepFace.extract_faces applies (cv2 rotation by the eye-line angle), but done on
    whatever image is passed in, so callers can align at full resolution.
    Only a window around the face is rotated, not the whole frame.
    """
    x, y, w, h = [int(v) for v in box]
    img_h, img_w = image.shape[:2]
    cx, cy = x + w / 2.0, y + h / 2.0

    angle = 0.0
    if left_eye is not None and right_eye is not None:
        angle = float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))

    # Big enough that rotating the box about its centre never pulls in out-of-window pixels
    half = max(w, h)
    wx1, wy1 = max(0, int(cx - half)), max(0, int(cy - half))
    wx2, wy2 = min(img_w, int(cx + half)), min(img_h, int(cy + half))
    window = image[wy1:wy2, wx1:wx2]
    if window.size == 0:
        return None

    if abs(angle) > 0.5:
        M = cv2.getRotationMatrix2D((cx - wx1, cy - wy1), angle, 1.0)
        window = cv2.warpAffine(window, M, (window.shape[1], window.shape[0]),
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    fx1 = max(0, int(round(cx - wx1 - w / 2.0)))
    fy1 = max(0, int(round(cy - wy1 - h / 2.0)))
    crop = window[fy1:fy1 + h, fx1:fx1 + w]
    return crop.copy() if crop.size > 0 else None


# Face quality gates. An embedding of a near-black or smeared face mostly encodes the
# darkness/blur, not the person: under a backlit classroom webcam, such enrolment crops
# landed within 0.02-0.09 cosine distance of each other (threshold 0.40), so whoever
# stood in front of the camera "matched" one of them. Brightness/sharpness are measured
# on the central part of the face (eyes/nose/mouth) - the box edges hold background and
# the face-to-background silhouette, which against a bright window inflate both numbers -
# downscaled to 112x112 when larger, so a big close-up face isn't scored on its fine skin
# texture. Smaller faces are measured as-is: upscaling smooths away exactly the edges the
# Laplacian measures, which scored every ~60px face in a phone/drone classroom video at
# 2-12 (all "blurry") although the same crops scored 15-300 at native size and matched
# their students. On the backlit-webcam deployment's crops, usable faces scored
# sharpness 19-53 and the unusable ones 3.
# Enrolment is strict, because one bad sample poisons every later scan; recognition is
# lenient, because the distance threshold + margin still guard each individual match.
ENROLL_QUALITY = {'min_size': 80, 'min_brightness': 30, 'min_sharpness': 10}
RECOGNITION_QUALITY = {'min_size': 40, 'min_brightness': 20, 'min_sharpness': 5}


def face_quality(face_bgr, box=None, min_size=0, min_brightness=0, min_sharpness=0):
    """
    Returns None if the face crop is usable, else the reason it isn't:
    'too_small', 'too_dark' or 'blurry'.
    """
    if face_bgr is None or face_bgr.size == 0:
        return 'too_small'
    if box is not None and min(box[2], box[3]) < min_size:
        return 'too_small'

    h, w = face_bgr.shape[:2]
    center = face_bgr[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    if center.size == 0:
        return 'too_small'
    if min(center.shape[:2]) > 112:
        center = cv2.resize(center, (112, 112), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    if gray.mean() < min_brightness:
        return 'too_dark'
    if cv2.Laplacian(gray, cv2.CV_64F).var() < min_sharpness:
        return 'blurry'
    return None


def normalize_lighting(face_bgr):
    """
    Equalises contrast on the luminance channel only (CLAHE), so a backlit, underexposed
    face and a well-lit one of the same person land closer together in embedding space.
    Applied identically at enrolment and recognition time.
    """
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def extract_embedding_from_crop(face_bgr, model_name=None):
    """
    Extracts a face embedding from an aligned BGR face crop (detect_faces_dnn's 'aligned').
    The face is already located, so DeepFace models run with detector_backend='skip' -
    no re-detecting (and re-aligning), keeping registration-time and recognition-time
    embeddings identically framed.
    """
    if face_bgr is None or face_bgr.size == 0:
        return None

    model_name = model_name or DEFAULT_MODEL
    if model_name == 'SFace':
        with _MODEL_LOCK:
            return _recognizer().feature(face_bgr).flatten()

    if USE_LIGHTING_NORMALIZATION:
        face_bgr = normalize_lighting(face_bgr)

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


def align_stored_face(img):
    """
    The aligned crop to embed for a stored enrolment sample, or None if no face is found.
    Samples saved by this version already are ALIGNED_SIZE-square aligned crops. Older ones
    are tight, unaligned face crops (200x200): those are padded - the detector needs some
    context around a face - and run through detection and alignment again.
    """
    if img.shape[:2] == (ALIGNED_SIZE, ALIGNED_SIZE):
        return img
    pad = max(img.shape[:2]) // 3
    detections = detect_faces_dnn(cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE))
    if not detections:
        return None
    return max(detections, key=lambda d: d['box'][2] * d['box'][3])['aligned']


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


# Cosine-distance thresholds per model. Mirrored from DeepFace's own tuned values
# (deepface/config/threshold.py) so picking a threshold doesn't force-import deepface (and
# the TensorFlow chain behind it, 20-40s on CPU) just to construct a FaceRecognitionSystem.
# SFace is the exception: it runs on OpenCV's 5-landmark alignment here, not DeepFace's
# eye-line one, so it uses OpenCV's threshold for that pipeline (cosine similarity 0.363).
# On this deployment's footage, genuine matches had a median similarity of 0.57 and the
# closest stranger 0.24.
_COSINE_THRESHOLDS = {
    'VGG-Face': 0.68, 'Facenet': 0.40, 'Facenet512': 0.30, 'ArcFace': 0.68,
    'Dlib': 0.07, 'SFace': 1 - 0.363, 'OpenFace': 0.10, 'DeepFace': 0.23,
    'DeepID': 0.015, 'GhostFaceNet': 0.65, 'Buffalo_L': 0.55,
}

# A face is only matched if its closest student beats the runner-up student by at least
# this fraction of the threshold - otherwise two similar-looking enrolments flip-flop
# between scans and attendance lands on whichever happened to be marginally closer.
MARGIN_FRACTION = 0.15

_LEGACY_CIRCLE_CORNER_MAX = 3


def _is_legacy_circular_crop(img):
    """Crops saved before the circular-mask fix have pure-black corners outside the circle."""
    k = max(4, img.shape[0] // 20)
    corners = [img[:k, :k], img[:k, -k:], img[-k:, :k], img[-k:, -k:]]
    return all(c.mean() <= _LEGACY_CIRCLE_CORNER_MAX for c in corners)


class FaceRecognitionSystem:
    def __init__(self, dataset_dir='dataset', model_name=None, distance_threshold=None, margin=None):
        self.dataset_dir = dataset_dir
        self.model_name = model_name or DEFAULT_MODEL
        self.distance_threshold = (
            distance_threshold if distance_threshold is not None
            else _COSINE_THRESHOLDS.get(self.model_name, 0.40)
        )
        self.margin = margin if margin is not None else MARGIN_FRACTION * self.distance_threshold
        self.embeddings_cache = {}
        self._rebuild_needed = False
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.load_model()

    @property
    def embeddings_cache_path(self):
        return os.path.join(self.dataset_dir, 'embeddings_cache.pkl')

    def load_model(self):
        """
        Loads cached student embeddings from disk into memory. Embeddings from different
        models (or differently prepared crops) aren't comparable, so a cache built another
        way (or the old untagged format) is discarded and rebuilt from the stored face
        crops on first use.
        """
        self.embeddings_cache = {}
        self._rebuild_needed = False
        if os.path.exists(self.embeddings_cache_path):
            try:
                with open(self.embeddings_cache_path, 'rb') as f:
                    data = pickle.load(f)
                if (isinstance(data, dict) and data.get('model') == self.model_name
                        and data.get('pipeline') == EMBEDDING_PIPELINE):
                    self.embeddings_cache = data.get('embeddings', {})
                else:
                    self._rebuild_needed = True
            except Exception as e:
                print(f"Warning: could not load embeddings cache: {e}")
                self._rebuild_needed = True
        return True

    def _persist_cache(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        with open(self.embeddings_cache_path, 'wb') as f:
            pickle.dump({'model': self.model_name, 'pipeline': EMBEDDING_PIPELINE,
                         'embeddings': self.embeddings_cache}, f)

    def _student_dirs(self):
        if not os.path.isdir(self.dataset_dir):
            return []
        dirs = []
        for entry in os.listdir(self.dataset_dir):
            m = re.fullmatch(r'student_(\d+)', entry)
            path = os.path.join(self.dataset_dir, entry)
            if m and os.path.isdir(path):
                dirs.append((int(m.group(1)), path))
        return sorted(dirs)

    def rebuild_embeddings(self):
        """
        Recomputes every student's embeddings from their stored face crops with the current
        model. Unusable crops (near-black or smeared, or still carrying the old circular
        mask) are left out rather than embedded - they cause false matches. Returns the ids
        of students left with no usable sample, who need to be registered again.

        The bar here is the lenient recognition one, not the enrolment one: samples passed
        the enrolment gate when captured, measured on the native-resolution crop, and the
        stored copy is resized and re-encoded, which lowers its score - samples captured
        through the gate scored sharpness 7.8-8.2 once stored, under the enrolment bar of 10.
        The crops this filter exists for, saved before the gate, scored 3.
        """
        cache = {}
        needs_reenrolment = []
        for student_id, path in self._student_dirs():
            for fname in sorted(os.listdir(path)):
                if not fname.endswith(('.jpg', '.png')):
                    continue
                img = cv2.imread(os.path.join(path, fname))
                if img is None or _is_legacy_circular_crop(img):
                    continue
                if face_quality(img, min_brightness=RECOGNITION_QUALITY['min_brightness'],
                                min_sharpness=RECOGNITION_QUALITY['min_sharpness']):
                    continue
                aligned = align_stored_face(img)
                emb = extract_embedding_from_crop(aligned, self.model_name) if aligned is not None else None
                if emb is not None:
                    cache.setdefault(student_id, []).append(emb)
            if student_id not in cache:
                needs_reenrolment.append(student_id)

        self.embeddings_cache = cache
        self._rebuild_needed = False
        self._persist_cache()
        if needs_reenrolment:
            print(f"Face embeddings rebuilt with {self.model_name}; no usable face sample for "
                  f"student id(s) {needs_reenrolment} - register them again to be recognised.")
        return needs_reenrolment

    def _ensure_cache(self):
        if self._rebuild_needed:
            self.rebuild_embeddings()

    def warmup(self):
        """
        Loads the detector and embedding model (downloading their weights if needed) and
        rebuilds a stale embeddings cache now, at server startup, instead of lazily on the
        first real request - which otherwise looks hung while that happens (a DeepFace
        model's TensorFlow cold start alone takes 20-40s on CPU).
        """
        dummy = np.ones((300, 300, 3), dtype=np.uint8) * 127
        detect_faces_dnn(dummy)
        extract_embedding_from_crop(cv2.resize(dummy, (ALIGNED_SIZE, ALIGNED_SIZE)), self.model_name)
        self._ensure_cache()
        return True

    def save_student_face(self, student_id, face_crop_bgr, embedding=None):
        """
        Saves a student's aligned face crop (detect_faces_dnn's 'aligned') to
        dataset_dir/student_{student_id}/ and caches its embedding in memory + on disk,
        so recognition never has to recompute it from the stored image again.
        """
        self._ensure_cache()
        student_dir = os.path.join(self.dataset_dir, f"student_{student_id}")
        os.makedirs(student_dir, exist_ok=True)

        existing_files = [f for f in os.listdir(student_dir) if f.endswith(('.jpg', '.png'))]
        img_idx = len(existing_files) + 1
        img_path = os.path.join(student_dir, f"{img_idx}.jpg")

        if len(face_crop_bgr.shape) == 2:
            face_crop_bgr = cv2.cvtColor(face_crop_bgr, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(img_path, face_crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if embedding is None:
            embedding = extract_embedding_from_crop(face_crop_bgr, self.model_name)
        if embedding is not None:
            self.embeddings_cache.setdefault(student_id, []).append(np.asarray(embedding))
            self._persist_cache()

        return True

    def replace_student_faces(self, student_id, samples):
        """
        Makes `samples` - a list of (face_crop_bgr, embedding) - the student's complete
        enrolment, discarding any earlier face samples (used when re-registering a student
        whose original capture was poor).
        """
        self._ensure_cache()
        student_dir = os.path.join(self.dataset_dir, f"student_{student_id}")
        if os.path.isdir(student_dir):
            for fname in os.listdir(student_dir):
                if fname.endswith(('.jpg', '.png')):
                    os.remove(os.path.join(student_dir, fname))
        self.embeddings_cache.pop(student_id, None)
        for crop, emb in samples:
            self.save_student_face(student_id, crop, embedding=emb)
        return True

    def rank_students(self, embedding, exclude_student_id=None):
        """
        Cosine distance from `embedding` to each enrolled student's closest sample,
        as [(student_id, distance)] sorted nearest first.
        """
        self._ensure_cache()
        e = np.asarray(embedding, dtype=np.float64)
        norm_e = np.linalg.norm(e)
        if norm_e == 0:
            return []
        e = e / norm_e

        ranking = []
        for student_id, emb_list in self.embeddings_cache.items():
            if student_id == exclude_student_id or not emb_list:
                continue
            M = np.asarray(emb_list, dtype=np.float64)
            norms = np.linalg.norm(M, axis=1)
            valid = norms > 0
            if not valid.any():
                continue
            dists = 1.0 - (M[valid] @ e) / norms[valid]
            ranking.append((student_id, max(0.0, float(dists.min()))))
        ranking.sort(key=lambda kv: kv[1])
        return ranking

    def match_embedding(self, embedding):
        """
        Returns (student_id or None, distance, reason). reason is None on a match, else
        'no_match' (nobody within the threshold) or 'ambiguous' (the two closest students
        are too close to call).
        """
        ranking = self.rank_students(embedding)
        if not ranking:
            return None, 1.0, 'no_match'
        student_id, dist = ranking[0]
        if dist > self.distance_threshold:
            return None, dist, 'no_match'
        if len(ranking) > 1 and ranking[1][1] - dist < self.margin:
            return None, dist, 'ambiguous'
        return student_id, dist, None

    def build_enrollment(self, face_crops, exclude_student_id=None):
        """
        Embeds candidate enrolment crops (several frames of one student) and drops any that
        disagree with the majority - e.g. the largest face briefly being someone walking past.
        Also checks the face doesn't already belong to a different enrolled student.

        Returns (samples, conflicting_student_id): samples is [(crop, embedding)], and
        conflicting_student_id is set if those samples already match another student.
        """
        candidates = []
        for crop in face_crops:
            emb = extract_embedding_from_crop(crop, self.model_name)
            if emb is not None:
                candidates.append((crop, np.asarray(emb)))
        if not candidates:
            return [], None

        E = np.stack([e / (np.linalg.norm(e) or 1.0) for _, e in candidates])
        D = 1.0 - E @ E.T
        medoid = int(np.argmin(D.sum(axis=1)))
        samples = [candidates[i] for i in range(len(candidates)) if D[medoid, i] <= self.distance_threshold]

        for _, emb in samples:
            ranking = self.rank_students(emb, exclude_student_id=exclude_student_id)
            if ranking and ranking[0][1] <= self.distance_threshold:
                return samples, ranking[0][0]
        return samples, None

    def recognize_faces_batch(self, b64_or_image):
        """
        Detects every face in the given frame and matches each one against the cached
        student embeddings. Pure recognition - does not write attendance records.

        Returns a list of dicts:
        {'box': (x, y, w, h), 'student_id': int|None, 'distance': float, 'reason': str|None}
        where reason explains an unmatched face: 'too_small', 'too_dark', 'blurry',
        'no_match', 'ambiguous', or 'duplicate' (another face in the frame matched the
        same student more closely).
        """
        if isinstance(b64_or_image, str):
            image = base64_to_image(b64_or_image)
        else:
            image = b64_or_image

        if image is None:
            return []

        self._ensure_cache()
        detections = detect_faces_dnn(image)
        results = []

        for det in detections:
            rec = {'box': det['box'], 'student_id': None, 'distance': 1.0, 'reason': None}
            results.append(rec)

            rec['reason'] = face_quality(det['face'], det['box'], **RECOGNITION_QUALITY)
            if rec['reason']:
                continue

            embedding = extract_embedding_from_crop(det['aligned'], self.model_name)
            if embedding is None:
                rec['reason'] = 'no_match'
                continue

            student_id, dist, reason = self.match_embedding(embedding)
            rec.update(student_id=student_id, distance=float(dist), reason=reason)

        # One person can only be in one place: if two faces in the frame matched the same
        # student, only the closer one keeps the match.
        claimed = set()
        for rec in sorted((r for r in results if r['student_id'] is not None), key=lambda r: r['distance']):
            if rec['student_id'] in claimed:
                rec['student_id'] = None
                rec['reason'] = 'duplicate'
            else:
                claimed.add(rec['student_id'])

        for rec in results:
            rec['distance'] = round(rec['distance'], 4)
        return results
