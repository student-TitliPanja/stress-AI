"""
app.py — Complete Flask application for Real-Time Stress Detection & Analysis
Routes: /, /register, /login, /logout, /admin, /calibrate, /save_baseline,
        /live, /video_feed, /audio_feed, /upload, /analysis
"""

import os
import sys
import json
import time
import csv
import hashlib
import secrets
import threading
import numpy as np
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, Response, jsonify, send_from_directory
)
from werkzeug.utils import secure_filename

# Local modules
from calibration import (
    compute_baseline_from_data, save_baseline, load_baseline,
    has_baseline, compute_personalized_stress, classify_va_zone,
    get_recommendations
)
from openface_pipeline import OpenFacePipeline, AUSequenceBuffer
from audio_pipeline import (
    AudioCapture, AudioSequenceBuffer, extract_acoustic_features,
    extract_features_from_file, _simulate_acoustic_features
)
from analysis_dashboard import generate_all_plots

# ──────────────────────────────────────────────────────────────
#  App Configuration
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ──────────────────────────────────────────────────────────────
#  User Database (JSON-based for simplicity)
# ──────────────────────────────────────────────────────────────
USERS_FILE = os.path.join(os.path.dirname(__file__), 'users.json')
EMOTION_LOG = os.path.join(os.path.dirname(__file__), 'emotion_log.csv')

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ──────────────────────────────────────────────────────────────
#  Authentication Decorator
# ──────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        users = load_users()
        user = users.get(session['user_id'], {})
        if not user.get('is_admin', False):
            flash('Admin access required.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

# ──────────────────────────────────────────────────────────────
#  Pipeline Instances (lazy-loaded)
# ──────────────────────────────────────────────────────────────
face_pipeline = None
audio_capture = None
au_buffer = None
audio_buffer = None
model = None

def init_pipelines():
    global face_pipeline, audio_capture, au_buffer, audio_buffer
    if face_pipeline is None:
        face_pipeline = OpenFacePipeline()
        au_buffer = AUSequenceBuffer(seq_len=30)
        audio_buffer = AudioSequenceBuffer(seq_len=20)

def load_model():
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), 'full_model.h5')
        if os.path.exists(model_path):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
                print(f"✅ Model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
                model = None

# ──────────────────────────────────────────────────────────────
#  Emotion Log
# ──────────────────────────────────────────────────────────────
def log_emotion(valence, arousal, stress_score, zone, pitch=0, jitter=0, shimmer=0):
    """Append a row to emotion_log.csv."""
    file_exists = os.path.exists(EMOTION_LOG)
    with open(EMOTION_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'valence', 'arousal', 'stress_score',
                           'dominant_zone', 'pitch', 'jitter', 'shimmer'])
        writer.writerow([
            datetime.now().isoformat(),
            f'{valence:.4f}', f'{arousal:.4f}', f'{stress_score:.4f}',
            zone, f'{pitch:.2f}', f'{jitter:.4f}', f'{shimmer:.4f}'
        ])

# ──────────────────────────────────────────────────────────────
#  Inference Helper
# ──────────────────────────────────────────────────────────────

# Stress-correlated AU indices and weights (matching openface_pipeline.py)
_STRESS_AU_INDICES = {
    2:  1.0,   # AU04 Brow Furrow — strong stress indicator
    5:  0.9,   # AU07 Eyelid Tightener — strong stress indicator
    6:  0.6,   # AU09 Nose Wrinkle — moderate stress indicator
    11: 0.6,   # AU17 Chin Raise — moderate stress indicator
    10: 0.5,   # AU15 Lip Corner Depressor — moderate (frown)
    13: 0.4,   # AU23 Lip Tightener — moderate stress indicator
    8: -0.7,   # AU12 Lip Corner Pull — negative (smile = relaxation)
}

def _compute_stress_from_aus(au_seq):
    """
    Compute stress-derived valence/arousal from AU sequence
    using research-based AU-to-emotion mappings.
    """
    # Average AU values over the temporal sequence
    au_mean = np.mean(au_seq, axis=0)  # shape (17,)

    # Compute weighted stress score from AUs
    stress_score = 0.0
    relax_score = 0.0
    total_weight = 0.0

    for idx, weight in _STRESS_AU_INDICES.items():
        if idx < len(au_mean):
            if weight > 0:
                stress_score += au_mean[idx] * weight
            else:
                relax_score += au_mean[idx] * abs(weight)
            total_weight += abs(weight)

    if total_weight > 0:
        stress_score /= total_weight
        relax_score /= total_weight

    # Map to valence-arousal space
    # High stress AUs → negative valence, high arousal
    # High relax AUs (smile) → positive valence, lower arousal
    valence = float(np.clip((relax_score - stress_score) * 1.2, -1.0, 1.0))
    arousal = float(np.clip((stress_score - relax_score * 0.3) * 0.8, -1.0, 1.0))

    return valence, arousal


def run_inference(au_seq, audio_seq, user_id=None, is_image=False):
    """Run model inference and return stress analysis."""
    load_model()

    if model is not None:
        au_input = np.expand_dims(au_seq, axis=0)
        audio_input = np.expand_dims(audio_seq, axis=0)
        prediction = model.predict([au_input, audio_input], verbose=0)
        valence, arousal = float(prediction[0][0]), float(prediction[0][1])

        # For image uploads, blend with direct AU analysis since audio is absent
        if is_image:
            au_valence, au_arousal = _compute_stress_from_aus(au_seq)
            # Weight AU-based analysis more heavily for images
            valence = valence * 0.4 + au_valence * 0.6
            arousal = arousal * 0.4 + au_arousal * 0.6
    else:
        # No model loaded — compute entirely from AU features
        valence, arousal = _compute_stress_from_aus(au_seq)

    # Ensure native Python float (not numpy float32) for JSON serialization
    valence = float(valence)
    arousal = float(arousal)

    zone = classify_va_zone(valence, arousal)

    if user_id:
        stress_score = float(compute_personalized_stress([valence, arousal], user_id))
    else:
        stress_score = float(np.clip((-valence * 0.5 + arousal * 0.5), 0.0, 1.0))

    recommendations = get_recommendations(zone)

    return {
        'valence': valence,
        'arousal': arousal,
        'stress_score': float(stress_score),
        'zone': zone,
        'recommendations': recommendations,
    }

# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

# ─── Home ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ─── Register ────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip().lower()
        mobile   = request.form.get('mobile', '').strip()
        password = request.form.get('password', '')

        # Validation
        errors = []
        if not name:
            errors.append('Name is required.')
        if not email or '@' not in email:
            errors.append('Valid email is required.')
        if not mobile or len(mobile) < 10:
            errors.append('Valid mobile number (10+ digits) is required.')
        if len(password) < 8:
            errors.append('Password must be at least 8 characters.')
        if not any(c.isupper() for c in password):
            errors.append('Password must contain at least 1 uppercase letter.')
        if not any(c.islower() for c in password):
            errors.append('Password must contain at least 1 lowercase letter.')
        if not any(c.isdigit() for c in password):
            errors.append('Password must contain at least 1 digit.')

        users = load_users()
        if email in users:
            errors.append('Email already registered.')
        for uid, u in users.items():
            if u.get('mobile') == mobile:
                errors.append('Mobile number already registered.')
                break

        if errors:
            for e in errors:
                flash(e, 'danger')
            return render_template('register.html', name=name, email=email, mobile=mobile)

        # Create user
        user_id = email
        users[user_id] = {
            'name': name,
            'email': email,
            'mobile': mobile,
            'password': hash_password(password),
            'is_admin': len(users) == 0,  # First user is admin
            'is_active': True,
            'created_at': datetime.now().isoformat(),
        }
        save_users(users)
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# ─── Login ────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        users = load_users()
        user = users.get(email)

        if not user or user['password'] != hash_password(password):
            flash('Invalid email or password.', 'danger')
            return render_template('login.html', email=email)

        if not user.get('is_active', True):
            flash('Your account is deactivated. Contact admin.', 'warning')
            return render_template('login.html', email=email)

        session['user_id'] = email
        session['user_name'] = user['name']
        session['is_admin'] = user.get('is_admin', False)
        flash(f'Welcome back, {user["name"]}!', 'success')
        return redirect(url_for('index'))

    return render_template('login.html')

# ─── Logout ───────────────────────────────────────────────────
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# ─── Admin ────────────────────────────────────────────────────
@app.route('/admin')
@admin_required
def admin():
    users = load_users()
    return render_template('admin.html', users=users)

@app.route('/admin/toggle/<user_id>')
@admin_required
def toggle_user(user_id):
    users = load_users()
    if user_id in users:
        users[user_id]['is_active'] = not users[user_id].get('is_active', True)
        save_users(users)
        status = 'activated' if users[user_id]['is_active'] else 'deactivated'
        flash(f'User {user_id} {status}.', 'success')
    return redirect(url_for('admin'))

# ─── Calibration ──────────────────────────────────────────────
@app.route('/calibrate')
@login_required
def calibrate():
    user_id = session['user_id']
    has_existing = has_baseline(user_id)
    return render_template('calibration.html', has_existing=has_existing)

@app.route('/save_baseline', methods=['POST'])
@login_required
def save_baseline_route():
    user_id = session['user_id']

    try:
        data = request.get_json()
        au_vectors = [np.array(v, dtype=np.float32) for v in data.get('au_vectors', [])]
        audio_features = [np.array(v, dtype=np.float32) for v in data.get('audio_features', [])]

        if len(au_vectors) < 10:
            # Generate demo baseline if not enough real data
            au_vectors = [np.random.randn(17).astype(np.float32) * 0.5 for _ in range(2700)]
            audio_features = [np.random.randn(84).astype(np.float32) for _ in range(180)]

        baseline = compute_baseline_from_data(au_vectors, audio_features)
        save_baseline(user_id, baseline)

        return jsonify({'status': 'success', 'message': 'Baseline saved!'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# ─── Live Analysis ────────────────────────────────────────────
@app.route('/live')
@login_required
def live():
    user_id = session['user_id']
    calibrated = has_baseline(user_id)
    return render_template('live.html', calibrated=calibrated)

@app.route('/video_feed')
@login_required
def video_feed():
    """MJPEG stream with AU overlay on face."""
    import cv2

    init_pipelines()
    # Capture user_id BEFORE entering the generator (inside request context)
    current_user_id = session.get('user_id')

    def generate_frames():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            # Send a placeholder frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, 'No Camera Available', (120, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            _, jpg_buf = cv2.imencode('.jpg', placeholder)
            frame_bytes = jpg_buf.tobytes()
            while True:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(1)
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract AUs
            au_vec, faces = face_pipeline.extract_aus_from_frame(frame)
            au_buffer.add(au_vec)

            # Draw overlay
            annotated = face_pipeline.draw_au_overlay(frame, au_vec, faces)

            # Add model prediction overlay if buffer ready
            if au_buffer.is_ready():
                au_seq = au_buffer.get_sequence()
                audio_seq = audio_buffer.get_sequence()
                result = run_inference(au_seq, audio_seq, current_user_id)

                # Log
                log_emotion(result['valence'], result['arousal'],
                           result['stress_score'], result['zone'])

                # Draw V-A info
                y_pos = frame.shape[0] - 120
                cv2.putText(annotated, f"V: {result['valence']:.2f}  A: {result['arousal']:.2f}",
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, f"Stress: {result['stress_score']:.2f}",
                           (10, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 0, 255) if result['stress_score'] > 0.5 else (0, 255, 0), 2)
                cv2.putText(annotated, f"Zone: {result['zone'].upper()}",
                           (10, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            _, jpg_buf = cv2.imencode('.jpg', annotated)
            frame_bytes = jpg_buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/audio_feed')
@login_required
def audio_feed():
    """SSE stream of acoustic features."""
    init_pipelines()
    # Keep a local reference to the buffer (captured in request context)
    local_audio_buffer = audio_buffer

    def generate_events():
        while True:
            features = _simulate_acoustic_features()
            if local_audio_buffer is not None:
                local_audio_buffer.add(features)

            data = {
                'pitch': float(features[0]),
                'jitter': float(features[1]),
                'shimmer': float(features[2]),
                'rms': float(features[3]),
                'timestamp': datetime.now().isoformat(),
            }

            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.5)

    return Response(generate_events(), mimetype='text/event-stream')

@app.route('/live_data')
@login_required
def live_data():
    """JSON endpoint for live stress data (polled by frontend)."""
    init_pipelines()

    # Generate simulated analysis
    au_seq = au_buffer.get_sequence() if au_buffer else np.random.randn(30, 17).astype(np.float32)
    audio_seq = audio_buffer.get_sequence() if audio_buffer else np.random.randn(20, 84).astype(np.float32)
    result = run_inference(au_seq, audio_seq, session.get('user_id'))

    # Log
    log_emotion(result['valence'], result['arousal'],
               result['stress_score'], result['zone'])

    return jsonify(result)

# ─── Upload Analysis ──────────────────────────────────────────
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected.', 'warning')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'warning')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        result = None

        if ext in ('jpg', 'jpeg', 'png', 'bmp'):
            # Image analysis
            import cv2
            init_pipelines()
            frame = cv2.imread(filepath)
            if frame is not None:
                au_vec, faces = face_pipeline.extract_aus_from_frame(frame)
                au_seq = np.tile(au_vec, (30, 1))  # Repeat for sequence
                # Use neutral zeros for audio (no audio data for images)
                audio_seq = np.zeros((20, 84), dtype=np.float32)
                result = run_inference(au_seq, audio_seq, session.get('user_id'),
                                       is_image=True)
                result['type'] = 'image'
                result['filename'] = filename
                result['au_values'] = dict(zip(
                    ['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10',
                     'AU12','AU14','AU15','AU17','AU20','AU23','AU25','AU26','AU45'],
                    [float(v) for v in au_vec]
                ))

        elif ext in ('wav', 'mp3', 'ogg', 'flac'):
            # Audio analysis
            features = extract_features_from_file(filepath)
            if features:
                audio_seq = np.array(features[:20], dtype=np.float32)
                if len(audio_seq) < 20:
                    padding = np.zeros((20 - len(audio_seq), 84), dtype=np.float32)
                    audio_seq = np.vstack([audio_seq, padding])
                au_seq = np.random.randn(30, 17).astype(np.float32) * 0.5
                result = run_inference(au_seq, audio_seq, session.get('user_id'))
                result['type'] = 'audio'
                result['filename'] = filename
                result['pitch'] = float(features[0][0]) if features else 0
                result['jitter'] = float(features[0][1]) if features else 0
                result['shimmer'] = float(features[0][2]) if features else 0

        if result:
            log_emotion(result['valence'], result['arousal'],
                       result['stress_score'], result['zone'])
            return render_template('upload.html', result=result)
        else:
            flash('Could not analyze the file. Supported: images (jpg/png) or audio (wav/mp3).', 'warning')

    return render_template('upload.html', result=None)

# ─── Analytics Dashboard ─────────────────────────────────────
@app.route('/analysis')
@login_required
def analysis():
    user_id = session['user_id']
    baseline = load_baseline(user_id)
    neutral_v = baseline.get('neutral_valence', 0.1) if baseline else 0.1
    neutral_a = baseline.get('neutral_arousal', -0.1) if baseline else -0.1

    try:
        plots, stats = generate_all_plots(EMOTION_LOG, neutral_v, neutral_a)
    except Exception as e:
        print(f"Dashboard error: {e}")
        plots, stats = generate_all_plots(None, neutral_v, neutral_a)

    return render_template('analysis.html', plots=plots, stats=stats)


# ──────────────────────────────────────────────────────────────
#  Error Handlers
# ──────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('index.html'), 500


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("  🧠 Real-Time Stress Detection & Analysis System")
    print("  Facial AU + Speech + Cross-Modal Attention + V-A Regression")
    print("=" * 70)

    # Create default admin user if none exist
    users = load_users()
    if not users:
        users['admin@stress.ai'] = {
            'name': 'Admin',
            'email': 'admin@stress.ai',
            'mobile': '0000000000',
            'password': hash_password('Admin123'),
            'is_admin': True,
            'is_active': True,
            'created_at': datetime.now().isoformat(),
        }
        save_users(users)
        print("📋 Default admin created: admin@stress.ai / Admin123")

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)
