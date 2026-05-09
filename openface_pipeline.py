"""
openface_pipeline.py — Wrapper for OpenFace 3.0 AU extraction
Extracts 17 AU intensity vectors per frame from webcam or image.
Falls back to landmark-based estimation if OpenFace is not installed.
"""

import os
import cv2
import numpy as np
import subprocess
import csv
import tempfile

# Try to import dlib for landmark-based AU estimation
try:
    import dlib
    _DLIB_AVAILABLE = True
except ImportError:
    _DLIB_AVAILABLE = False

# Path to dlib's 68-point face landmark predictor
_LANDMARK_PREDICTOR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'shape_predictor_68_face_landmarks.dat'
)
_landmark_predictor = None
_dlib_detector = None

def _init_dlib():
    """Lazy-load dlib face detector and landmark predictor."""
    global _landmark_predictor, _dlib_detector
    if not _DLIB_AVAILABLE:
        return False
    if _landmark_predictor is not None:
        return True
    if os.path.exists(_LANDMARK_PREDICTOR_PATH):
        _dlib_detector = dlib.get_frontal_face_detector()
        _landmark_predictor = dlib.shape_predictor(_LANDMARK_PREDICTOR_PATH)
        print("✅ dlib landmark predictor loaded for AU estimation")
        return True
    else:
        print(f"⚠️  dlib predictor not found at {_LANDMARK_PREDICTOR_PATH}")
        print("   Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return False

# The 17 Action Units extracted by OpenFace 3.0
AU_NAMES = [
    'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09',
    'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23',
    'AU25', 'AU26', 'AU45'
]

# Stress-correlated AUs and their weights
STRESS_AU_WEIGHTS = {
    'AU04': 1.0,   # Brow Furrow — strong positive
    'AU07': 0.9,   # Eyelid Tightener — strong positive
    'AU09': 0.6,   # Nose Wrinkle — moderate positive
    'AU17': 0.6,   # Chin Raise — moderate positive
    'AU12': -0.5,  # Lip Corner Pull — negative (relaxation indicator)
}

class OpenFacePipeline:
    """
    Wraps OpenFace 3.0 for real-time AU extraction.
    If OpenFace binary is not available, uses a simulation mode
    that generates plausible AU intensities from face landmarks.
    """

    def __init__(self, openface_path=None):
        """
        Args:
            openface_path: Path to OpenFace FeatureExtraction binary.
                           If None, uses simulation mode.
        """
        self.openface_path = openface_path
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.simulation_mode = openface_path is None or not os.path.exists(str(openface_path))

        if self.simulation_mode:
            print("⚠️  OpenFace not found — running in simulation mode")
            print("   Install OpenFace 3.0 and set openface_path for real AU extraction")
        else:
            print(f"✅ OpenFace found at: {openface_path}")

    def detect_faces(self, frame):
        """Detect faces using Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces

    def extract_aus_from_frame(self, frame):
        """
        Extract 17-dim AU intensity vector from a single frame.

        Returns:
            au_vector: np.ndarray of shape (17,) — AU intensities [0, 5]
            faces: detected face bounding boxes
        """
        faces = self.detect_faces(frame)

        if len(faces) == 0:
            return np.zeros(17, dtype=np.float32), []

        if self.simulation_mode:
            return self._simulate_aus(frame, faces[0]), faces
        else:
            return self._extract_aus_openface(frame, faces[0]), faces

    def _simulate_aus(self, frame, face_bbox):
        """
        Estimate AU intensities when OpenFace is not available.
        Uses dlib 68-point landmarks for geometric AU estimation,
        falls back to enhanced pixel analysis if dlib unavailable.
        """
        # Try landmark-based approach first
        if _init_dlib():
            au_vec = self._estimate_aus_from_landmarks(frame, face_bbox)
            if au_vec is not None:
                return au_vec

        # Fallback: enhanced pixel-based estimation
        return self._estimate_aus_from_pixels(frame, face_bbox)

    def _estimate_aus_from_landmarks(self, frame, face_bbox):
        """
        Estimate AUs from dlib 68-point face landmarks using
        anatomically meaningful geometric relationships.
        """
        try:
            x, y, w, h = face_bbox
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = _landmark_predictor(gray, rect)
            pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)

            au_vector = np.zeros(17, dtype=np.float32)

            # Helper: Euclidean distance
            def dist(a, b):
                return np.linalg.norm(pts[a] - pts[b])

            # Normalize distances by inter-ocular distance
            iod = dist(36, 45)  # left eye outer corner to right eye outer corner
            if iod < 1.0:
                return None

            # --- Brow AUs ---
            # AU01 Inner Brow Raise: distance from inner brow (21,22) to eye corners
            left_inner_brow_height = (pts[21][1] - pts[39][1]) / iod  # brow to eye
            right_inner_brow_height = (pts[22][1] - pts[42][1]) / iod
            inner_brow_raise = -((left_inner_brow_height + right_inner_brow_height) / 2)
            au_vector[0] = np.clip(inner_brow_raise * 8.0, 0.0, 5.0)  # AU01

            # AU02 Outer Brow Raise: outer brow (17,26) distance from eye corners
            left_outer_brow_height = (pts[17][1] - pts[36][1]) / iod
            right_outer_brow_height = (pts[26][1] - pts[45][1]) / iod
            outer_brow_raise = -((left_outer_brow_height + right_outer_brow_height) / 2)
            au_vector[1] = np.clip(outer_brow_raise * 7.0, 0.0, 5.0)  # AU02

            # AU04 Brow Furrow: distance between inner brows (21, 22) relative to IOD
            brow_distance = dist(21, 22) / iod
            # Smaller distance = more furrowing; typical neutral ~0.25-0.35
            brow_furrow = np.clip((0.35 - brow_distance) * 15.0, 0.0, 5.0)
            au_vector[2] = brow_furrow  # AU04

            # --- Eye AUs ---
            # AU05 Upper Lid Raise: eye aspect ratio (height/width)
            left_ear = (dist(37, 41) + dist(38, 40)) / (2.0 * dist(36, 39))
            right_ear = (dist(43, 47) + dist(44, 46)) / (2.0 * dist(42, 45))
            avg_ear = (left_ear + right_ear) / 2.0
            au_vector[3] = np.clip((avg_ear - 0.25) * 20.0, 0.0, 5.0)  # AU05

            # AU06 Cheek Raise: cheek points moving up (detect via under-eye region)
            cheek_raise_left = (pts[36][1] - pts[1][1]) / iod
            cheek_raise_right = (pts[45][1] - pts[15][1]) / iod
            au_vector[4] = np.clip(-(cheek_raise_left + cheek_raise_right) * 5.0, 0.0, 5.0)  # AU06

            # AU07 Eyelid Tightener: low eye aperture without blink
            lid_tightness = np.clip((0.28 - avg_ear) * 18.0, 0.0, 5.0)
            au_vector[5] = lid_tightness  # AU07

            # AU45 Blink: very low eye aspect ratio
            au_vector[16] = np.clip((0.20 - avg_ear) * 25.0, 0.0, 5.0)  # AU45

            # --- Nose AU ---
            # AU09 Nose Wrinkle: detect via nostril width vs nose bridge width
            nostril_width = dist(31, 35) / iod
            nose_bridge_width = dist(21, 22) / iod
            au_vector[6] = np.clip((nostril_width - nose_bridge_width) * 12.0, 0.0, 5.0)  # AU09

            # AU10 Upper Lip Raiser: distance from nose bottom to upper lip
            nose_to_lip = dist(33, 51) / iod
            au_vector[7] = np.clip((0.35 - nose_to_lip) * 15.0, 0.0, 5.0)  # AU10

            # --- Mouth AUs ---
            mouth_width = dist(48, 54) / iod
            mouth_height = dist(51, 57) / iod
            mouth_aspect = mouth_height / (mouth_width + 1e-6)

            # Lip corner positions relative to neutral
            left_corner_y = (pts[48][1] - pts[33][1]) / iod
            right_corner_y = (pts[54][1] - pts[33][1]) / iod
            avg_corner_y = (left_corner_y + right_corner_y) / 2.0

            # AU12 Lip Corner Pull (smile): corners up, wider mouth
            smile_indicator = np.clip(-avg_corner_y * 8.0 + (mouth_width - 0.55) * 5.0, 0.0, 5.0)
            au_vector[8] = smile_indicator  # AU12

            # AU14 Dimpler: lip corners pulled back laterally
            au_vector[9] = np.clip((mouth_width - 0.55) * 6.0, 0.0, 5.0)  # AU14

            # AU15 Lip Corner Depressor (frown): corners down
            frown_indicator = np.clip(avg_corner_y * 10.0, 0.0, 5.0)
            au_vector[10] = frown_indicator  # AU15

            # AU17 Chin Raise: chin boss pushed up
            chin_to_lip = dist(57, 8) / iod
            au_vector[11] = np.clip((0.45 - chin_to_lip) * 12.0, 0.0, 5.0)  # AU17

            # AU20 Lip Stretcher: mouth wider than normal
            au_vector[12] = np.clip((mouth_width - 0.60) * 10.0, 0.0, 5.0)  # AU20

            # AU23 Lip Tightener: lips pressed together, small opening
            inner_lip_height = (dist(61, 67) + dist(62, 66)) / (2.0 * iod)
            au_vector[13] = np.clip((0.08 - inner_lip_height) * 30.0, 0.0, 5.0)  # AU23

            # AU25 Lips Part: mouth open
            au_vector[14] = np.clip(mouth_height * 8.0, 0.0, 5.0)  # AU25

            # AU26 Jaw Drop: large mouth opening
            jaw_drop = dist(62, 66) / iod
            au_vector[15] = np.clip(jaw_drop * 10.0, 0.0, 5.0)  # AU26

            return au_vector

        except Exception as e:
            print(f"Landmark AU estimation error: {e}")
            return None

    def _estimate_aus_from_pixels(self, frame, face_bbox):
        """
        Enhanced pixel-based AU estimation fallback.
        Uses gradient analysis and regional contrast ratios for better accuracy.
        """
        x, y, w, h = face_bbox
        face_roi = frame[y:y+h, x:x+w]

        if face_roi.size == 0:
            return np.zeros(17, dtype=np.float32)

        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.resize(gray_roi, (128, 128))

        # Apply histogram equalization for consistent lighting
        gray_roi = cv2.equalizeHist(gray_roi)

        # Compute edge maps for wrinkle/crease detection
        edges = cv2.Canny(gray_roi, 50, 150)
        sobel_h = np.abs(cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3))
        sobel_v = np.abs(cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3))

        # Define regions (128x128)
        brow_region = gray_roi[10:35, 15:113]
        eye_region = gray_roi[30:55, 15:113]
        nose_region = gray_roi[50:75, 35:93]
        mouth_region = gray_roi[75:115, 25:103]

        brow_edges = edges[10:35, 15:113]
        forehead_horiz = sobel_h[10:30, 15:113]
        glabellar_edges = edges[15:35, 45:83]  # Between brows

        au_vector = np.zeros(17, dtype=np.float32)

        # AU01/AU02: Brow raise → horizontal forehead wrinkles
        forehead_wrinkle_density = np.mean(forehead_horiz) / 50.0
        au_vector[0] = np.clip(forehead_wrinkle_density * 2.5, 0.0, 5.0)  # AU01
        au_vector[1] = np.clip(forehead_wrinkle_density * 2.0, 0.0, 5.0)  # AU02

        # AU04: Brow furrow → vertical wrinkles between brows (glabella)
        glabellar_density = np.mean(glabellar_edges) / 30.0
        au_vector[2] = np.clip(glabellar_density * 3.0, 0.0, 5.0)  # AU04

        # Eye-region intensity and gradient
        eye_gradient = np.mean(np.abs(cv2.Sobel(eye_region, cv2.CV_64F, 0, 1, ksize=3))) / 40.0
        eye_std = np.std(eye_region) / 60.0
        au_vector[3] = np.clip(eye_gradient * 1.5, 0.0, 5.0)  # AU05
        au_vector[4] = np.clip(eye_std * 1.2, 0.0, 5.0)       # AU06
        au_vector[5] = np.clip(eye_gradient * 2.0, 0.0, 5.0)  # AU07

        # AU45 Blink: detect if eye region is very dark/uniform
        eye_uniformity = 1.0 - (np.std(eye_region) / 80.0)
        au_vector[16] = np.clip(eye_uniformity * 2.0, 0.0, 5.0)  # AU45

        # Nose
        nose_creases = np.mean(edges[50:70, 30:98]) / 40.0
        au_vector[6] = np.clip(nose_creases * 2.0, 0.0, 5.0)  # AU09
        au_vector[7] = np.clip(nose_creases * 1.5, 0.0, 5.0)  # AU10

        # Mouth analysis
        mouth_edges = edges[75:115, 25:103]
        mouth_edge_density = np.mean(mouth_edges) / 35.0
        mouth_gradient_h = np.mean(sobel_h[75:115, 25:103]) / 40.0
        mouth_std = np.std(mouth_region) / 60.0

        # Detect mouth curvature from edge pattern
        upper_mouth_edges = np.mean(edges[75:90, 25:103])
        lower_mouth_edges = np.mean(edges[95:115, 25:103])
        mouth_asymmetry = abs(upper_mouth_edges - lower_mouth_edges) / 50.0

        au_vector[8]  = np.clip(mouth_edge_density * 1.8, 0.0, 5.0)      # AU12
        au_vector[9]  = np.clip(mouth_asymmetry * 2.0, 0.0, 5.0)         # AU14
        au_vector[10] = np.clip(mouth_gradient_h * 1.5, 0.0, 5.0)        # AU15
        au_vector[11] = np.clip(mouth_std * 1.8, 0.0, 5.0)               # AU17
        au_vector[12] = np.clip(mouth_edge_density * 1.2, 0.0, 5.0)      # AU20
        au_vector[13] = np.clip((1.0 - mouth_std) * 1.5, 0.0, 5.0)       # AU23
        au_vector[14] = np.clip(mouth_gradient_h * 2.0, 0.0, 5.0)        # AU25
        au_vector[15] = np.clip(mouth_edge_density * 1.0, 0.0, 5.0)      # AU26

        return au_vector

    def _extract_aus_openface(self, frame, face_bbox):
        """
        Extract AUs using the real OpenFace 3.0 binary.
        Writes frame to temp file, runs FeatureExtraction, parses CSV.
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                img_path = os.path.join(tmpdir, 'frame.jpg')
                out_dir  = os.path.join(tmpdir, 'output')
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(img_path, frame)

                # Run OpenFace
                cmd = [
                    self.openface_path,
                    '-f', img_path,
                    '-out_dir', out_dir,
                    '-aus',
                ]
                subprocess.run(cmd, capture_output=True, timeout=10)

                # Parse output CSV
                csv_path = os.path.join(out_dir, 'frame.csv')
                if os.path.exists(csv_path):
                    return self._parse_openface_csv(csv_path)

        except Exception as e:
            print(f"OpenFace error: {e}")

        return np.zeros(17, dtype=np.float32)

    def _parse_openface_csv(self, csv_path):
        """Parse OpenFace CSV output to extract AU intensity columns."""
        au_vector = np.zeros(17, dtype=np.float32)
        au_columns = [f' AU{name[2:]}_r' for name in AU_NAMES]

        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for i, col in enumerate(au_columns):
                        if col in row:
                            au_vector[i] = float(row[col])
                    break  # Only need first row for single frame
        except Exception as e:
            print(f"CSV parse error: {e}")

        return au_vector

    def compute_stress_indicator(self, au_vector):
        """
        Compute a quick stress indicator from AU weights.
        Higher value = more likely stressed.
        """
        score = 0.0
        au_dict = dict(zip(AU_NAMES, au_vector))

        for au_name, weight in STRESS_AU_WEIGHTS.items():
            if au_name in au_dict:
                score += au_dict[au_name] * weight

        return np.clip(score / 3.0, 0.0, 1.0)

    def draw_au_overlay(self, frame, au_vector, faces):
        """Draw AU values and face bounding box on frame."""
        annotated = frame.copy()

        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Draw stress indicator
            stress = self.compute_stress_indicator(au_vector)
            color = (0, int(255 * (1 - stress)), int(255 * stress))
            cv2.putText(annotated, f'Stress: {stress:.2f}',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw top stress-correlated AUs in corner
        y_offset = 30
        for i, (au_name, au_val) in enumerate(zip(AU_NAMES, au_vector)):
            if au_name in STRESS_AU_WEIGHTS:
                text = f'{au_name}: {au_val:.2f}'
                cv2.putText(annotated, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

        return annotated


class AUSequenceBuffer:
    """
    Buffers AU vectors into temporal sequences for LSTM input.
    Default: 30 frames = 1 second @ 30fps.
    """

    def __init__(self, seq_len=30, au_dim=17):
        self.seq_len = seq_len
        self.au_dim = au_dim
        self.buffer = []

    def add(self, au_vector):
        """Add a new AU vector to the buffer."""
        self.buffer.append(au_vector.copy())
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

    def is_ready(self):
        """Check if buffer has enough frames."""
        return len(self.buffer) >= self.seq_len

    def get_sequence(self):
        """Get the buffered sequence as a numpy array."""
        if not self.is_ready():
            # Pad with zeros if not enough frames
            padding = [np.zeros(self.au_dim)] * (self.seq_len - len(self.buffer))
            return np.array(padding + self.buffer, dtype=np.float32)
        return np.array(self.buffer[-self.seq_len:], dtype=np.float32)

    def reset(self):
        """Clear the buffer."""
        self.buffer = []


# ──────────────────────────────────────────────────────────────
#  Demo / Test
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  OpenFace Pipeline — Demo")
    print("=" * 60)

    pipeline = OpenFacePipeline()
    buffer = AUSequenceBuffer(seq_len=30)

    # Test with webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam. Testing with synthetic frame...")
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        au_vec, faces = pipeline.extract_aus_from_frame(test_frame)
        print(f"AU vector: {au_vec}")
        print(f"Stress indicator: {pipeline.compute_stress_indicator(au_vec):.3f}")
    else:
        print("📹 Webcam opened. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            au_vec, faces = pipeline.extract_aus_from_frame(frame)
            buffer.add(au_vec)

            annotated = pipeline.draw_au_overlay(frame, au_vec, faces)
            cv2.imshow('AU Extraction', annotated)

            if buffer.is_ready():
                seq = buffer.get_sequence()
                print(f"Sequence shape: {seq.shape}, Stress: {pipeline.compute_stress_indicator(au_vec):.3f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
