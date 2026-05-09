"""
audio_pipeline.py — Real-time audio capture + acoustic feature extraction
Extracts: Pitch (F0), Jitter, Shimmer, MFCCs (40), Delta MFCCs (40), RMS Energy
Total feature vector: 84 dimensions per 500ms chunk
"""

import numpy as np
import threading
import queue
import time

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa not installed — using simulation mode for audio features")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️  pyaudio not installed — using simulation mode for audio capture")


# ──────────────────────────────────────────────────────────────
#  Acoustic Feature Extraction (84 dimensions)
# ──────────────────────────────────────────────────────────────
def extract_acoustic_features(audio_chunk, sr=22050):
    """
    Extract 84-dimensional acoustic feature vector from a 500ms audio chunk.

    Features:
        - Pitch (F0 mean): 1 dim
        - Jitter: 1 dim
        - Shimmer: 1 dim
        - RMS Energy: 1 dim
        - MFCCs: 40 dims
        - Delta MFCCs: 40 dims
    Total: 84 dimensions

    Args:
        audio_chunk: np.ndarray of audio samples (float32)
        sr: sample rate (default 22050)

    Returns:
        np.ndarray of shape (84,)
    """
    if not LIBROSA_AVAILABLE:
        return _simulate_acoustic_features()

    try:
        # Ensure float32
        audio_chunk = audio_chunk.astype(np.float32)

        # ─── Pitch (F0) + Jitter ───
        f0, voiced_flag, _ = librosa.pyin(
            audio_chunk, fmin=75, fmax=300, sr=sr
        )
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
        pitch = np.nanmean(f0_clean) if len(f0_clean) > 0 else 0.0

        if len(f0_clean) > 1:
            jitter = np.mean(np.abs(np.diff(f0_clean)))
        else:
            jitter = 0.0

        # ─── Shimmer ───
        shimmer = np.mean(np.abs(np.diff(np.abs(audio_chunk))))

        # ─── MFCCs + Delta MFCCs (40 each) ───
        mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        mfcc_mean  = np.mean(mfcc, axis=1)   # (40,)
        delta_mean = np.mean(delta, axis=1)   # (40,)

        # ─── Energy (RMS) ───
        rms = np.mean(librosa.feature.rms(y=audio_chunk))

        # ─── Concatenate ───
        feature_vector = np.concatenate([
            [pitch, jitter, shimmer, rms],  # 4 dims
            mfcc_mean,                       # 40 dims
            delta_mean                       # 40 dims
        ])  # Total: 84

        # Handle NaN/Inf
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_vector.astype(np.float32)

    except Exception as e:
        print(f"Audio feature extraction error: {e}")
        return np.zeros(84, dtype=np.float32)


def _simulate_acoustic_features():
    """Generate simulated acoustic features when librosa is not available."""
    pitch   = np.random.uniform(80, 250)
    jitter  = np.random.uniform(0, 0.05)
    shimmer = np.random.uniform(0, 0.1)
    rms     = np.random.uniform(0.01, 0.3)
    mfcc    = np.random.randn(40) * 5
    delta   = np.random.randn(40) * 2

    return np.concatenate([[pitch, jitter, shimmer, rms], mfcc, delta]).astype(np.float32)


# ──────────────────────────────────────────────────────────────
#  Real-Time Audio Capture
# ──────────────────────────────────────────────────────────────
class AudioCapture:
    """
    Captures audio from microphone in real-time using PyAudio.
    Buffers 500ms chunks and extracts acoustic features.
    """

    def __init__(self, sr=22050, chunk_duration=0.5, device_index=None):
        """
        Args:
            sr: Sample rate
            chunk_duration: Duration of each chunk in seconds (0.5 = 500ms)
            device_index: PyAudio device index (None = default mic)
        """
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sr * chunk_duration)
        self.device_index = device_index

        self.audio_queue = queue.Queue(maxsize=100)
        self.feature_queue = queue.Queue(maxsize=100)
        self.is_recording = False
        self._stream = None
        self._pa = None
        self._thread = None

    def start(self):
        """Start recording audio in a background thread."""
        if not PYAUDIO_AVAILABLE:
            print("⚠️  PyAudio not available — starting simulation mode")
            self.is_recording = True
            self._thread = threading.Thread(target=self._simulate_recording, daemon=True)
            self._thread.start()
            return

        try:
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sr,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.is_recording = True
            self._stream.start_stream()
            print(f"🎤 Audio capture started (SR={self.sr}, chunk={self.chunk_duration}s)")

            # Start feature extraction thread
            self._thread = threading.Thread(target=self._process_audio, daemon=True)
            self._thread.start()

        except Exception as e:
            print(f"❌ Audio capture failed: {e}")
            print("   Falling back to simulation mode")
            self.is_recording = True
            self._thread = threading.Thread(target=self._simulate_recording, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop recording."""
        self.is_recording = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()
        print("🛑 Audio capture stopped")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback — pushes raw audio to queue."""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            pass  # Drop oldest if queue is full
        return (in_data, pyaudio.paContinue)

    def _process_audio(self):
        """Process audio chunks and extract features."""
        while self.is_recording:
            try:
                audio_chunk = self.audio_queue.get(timeout=1.0)
                features = extract_acoustic_features(audio_chunk, self.sr)
                try:
                    self.feature_queue.put_nowait(features)
                except queue.Full:
                    # Remove oldest and add new
                    try:
                        self.feature_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.feature_queue.put_nowait(features)
            except queue.Empty:
                continue

    def _simulate_recording(self):
        """Simulate audio recording when PyAudio is not available."""
        while self.is_recording:
            features = _simulate_acoustic_features()
            try:
                self.feature_queue.put_nowait(features)
            except queue.Full:
                try:
                    self.feature_queue.get_nowait()
                except queue.Empty:
                    pass
                self.feature_queue.put_nowait(features)
            time.sleep(self.chunk_duration)

    def get_latest_features(self):
        """Get the most recent acoustic feature vector."""
        latest = None
        while not self.feature_queue.empty():
            try:
                latest = self.feature_queue.get_nowait()
            except queue.Empty:
                break
        return latest


class AudioSequenceBuffer:
    """
    Buffers acoustic feature vectors into sequences for LSTM input.
    Default: 20 × 500ms chunks = 10 seconds of audio.
    """

    def __init__(self, seq_len=20, feat_dim=84):
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.buffer = []

    def add(self, feature_vector):
        """Add a new feature vector to the buffer."""
        self.buffer.append(feature_vector.copy())
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

    def is_ready(self):
        """Check if buffer has enough chunks."""
        return len(self.buffer) >= self.seq_len

    def get_sequence(self):
        """Get the buffered sequence as a numpy array."""
        if not self.is_ready():
            padding = [np.zeros(self.feat_dim)] * (self.seq_len - len(self.buffer))
            return np.array(padding + self.buffer, dtype=np.float32)
        return np.array(self.buffer[-self.seq_len:], dtype=np.float32)

    def reset(self):
        """Clear the buffer."""
        self.buffer = []


# ──────────────────────────────────────────────────────────────
#  Extract features from an audio file
# ──────────────────────────────────────────────────────────────
def extract_features_from_file(filepath, sr=22050, chunk_duration=0.5):
    """
    Extract acoustic features from an audio file.

    Returns:
        List of 84-dim feature vectors, one per 500ms chunk
    """
    if not LIBROSA_AVAILABLE:
        print("⚠️  librosa not available — returning simulated features")
        return [_simulate_acoustic_features() for _ in range(20)]

    try:
        audio, file_sr = librosa.load(filepath, sr=sr)
        chunk_size = int(sr * chunk_duration)
        features = []

        for start in range(0, len(audio) - chunk_size, chunk_size):
            chunk = audio[start:start + chunk_size]
            feat = extract_acoustic_features(chunk, sr)
            features.append(feat)

        return features

    except Exception as e:
        print(f"Error processing audio file: {e}")
        return [np.zeros(84, dtype=np.float32)]


# ──────────────────────────────────────────────────────────────
#  Demo / Test
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Audio Pipeline — Demo")
    print("=" * 60)

    # Test feature extraction with synthetic audio
    print("\n📊 Testing feature extraction...")
    test_audio = np.random.randn(11025).astype(np.float32) * 0.1  # 0.5s @ 22050Hz
    features = extract_acoustic_features(test_audio)
    print(f"Feature vector shape: {features.shape}")
    print(f"Pitch: {features[0]:.2f} Hz")
    print(f"Jitter: {features[1]:.4f}")
    print(f"Shimmer: {features[2]:.4f}")
    print(f"RMS Energy: {features[3]:.4f}")
    print(f"MFCC[0]: {features[4]:.2f}")

    # Test sequence buffer
    print("\n📦 Testing sequence buffer...")
    buffer = AudioSequenceBuffer(seq_len=20)
    for i in range(25):
        buffer.add(_simulate_acoustic_features())
    seq = buffer.get_sequence()
    print(f"Sequence shape: {seq.shape}")

    # Test real-time capture (5 seconds)
    print("\n🎤 Testing real-time capture (5 seconds)...")
    capture = AudioCapture()
    capture.start()
    time.sleep(5)
    capture.stop()

    count = 0
    while not capture.feature_queue.empty():
        feat = capture.feature_queue.get()
        count += 1
    print(f"Captured {count} feature chunks in 5 seconds")

    print("\n✅ Audio pipeline test complete!")
