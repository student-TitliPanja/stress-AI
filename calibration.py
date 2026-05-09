"""
calibration.py — 90-second neutral baseline recording, averaging, and local JSON storage.
Records the user's AU vectors and acoustic features during a neutral state
to compute a personalized baseline for stress deviation detection.
"""

import os
import json
import time
import numpy as np
from datetime import datetime


BASELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baselines')


def ensure_baseline_dir():
    """Create baselines directory if it doesn't exist."""
    os.makedirs(BASELINE_DIR, exist_ok=True)


def compute_baseline_from_data(au_vectors, audio_features):
    """
    Compute the personal neutral baseline from collected calibration data.

    Args:
        au_vectors: list of 17-dim AU vectors collected during calibration
        audio_features: list of 84-dim audio feature vectors

    Returns:
        dict with baseline statistics
    """
    au_array = np.array(au_vectors, dtype=np.float32)
    audio_array = np.array(audio_features, dtype=np.float32)

    baseline = {
        # AU statistics
        'au_mean': au_array.mean(axis=0).tolist(),
        'au_std': au_array.std(axis=0).tolist(),

        # Audio statistics
        'audio_mean': audio_array.mean(axis=0).tolist(),
        'audio_std': audio_array.std(axis=0).tolist(),

        # Key features for quick stress check
        'neutral_pitch': float(audio_array[:, 0].mean()),
        'neutral_pitch_std': float(audio_array[:, 0].std()),
        'neutral_jitter': float(audio_array[:, 1].mean()),
        'neutral_shimmer': float(audio_array[:, 2].mean()),
        'neutral_rms': float(audio_array[:, 3].mean()),

        # Stress-correlated AU baselines
        'neutral_au04': float(au_array[:, 2].mean()),  # Brow Furrow
        'neutral_au07': float(au_array[:, 5].mean()),  # Eyelid Tightener
        'neutral_au09': float(au_array[:, 6].mean()),  # Nose Wrinkle
        'neutral_au17': float(au_array[:, 11].mean()), # Chin Raise
        'neutral_au12': float(au_array[:, 8].mean()),  # Lip Corner Pull

        # V-A neutral point (estimated from feature statistics)
        'neutral_valence': 0.1,   # Default neutral valence
        'neutral_arousal': -0.1,  # Default neutral arousal

        # Metadata
        'calibration_duration_sec': 90,
        'num_au_samples': len(au_vectors),
        'num_audio_samples': len(audio_features),
        'calibrated_at': datetime.now().isoformat(),
    }

    return baseline


def save_baseline(user_id, baseline_data):
    """
    Save the personal baseline as a local JSON file.

    Args:
        user_id: unique user identifier
        baseline_data: dict with baseline statistics
    """
    ensure_baseline_dir()
    filepath = os.path.join(BASELINE_DIR, f'{user_id}_baseline.json')

    with open(filepath, 'w') as f:
        json.dump(baseline_data, f, indent=2)

    print(f"💾 Baseline saved for user '{user_id}' at {filepath}")
    return filepath


def load_baseline(user_id):
    """
    Load the personal baseline for a user.

    Returns:
        dict with baseline data, or None if not found
    """
    filepath = os.path.join(BASELINE_DIR, f'{user_id}_baseline.json')

    if not os.path.exists(filepath):
        print(f"⚠️  No baseline found for user '{user_id}'")
        return None

    with open(filepath, 'r') as f:
        return json.load(f)


def has_baseline(user_id):
    """Check if a user has a saved baseline."""
    filepath = os.path.join(BASELINE_DIR, f'{user_id}_baseline.json')
    return os.path.exists(filepath)


def compute_personalized_stress(va_output, user_id):
    """
    Compute personalized stress score by comparing V-A output
    against the user's calibrated neutral baseline.

    A stress score of 0.0 = calm/normal
    A stress score of 1.0 = maximum stress deviation

    Args:
        va_output: [valence, arousal] prediction from model
        user_id: user identifier for loading baseline

    Returns:
        float stress_score in [0, 1]
    """
    baseline = load_baseline(user_id)

    if baseline is None:
        # No calibration — use default neutral
        neutral_v = 0.1
        neutral_a = -0.1
    else:
        neutral_v = baseline.get('neutral_valence', 0.1)
        neutral_a = baseline.get('neutral_arousal', -0.1)

    valence = float(va_output[0])
    arousal = float(va_output[1])

    # Stress = deviation from neutral
    # Lower valence from neutral = more negative = more stressed
    delta_v = neutral_v - valence  # positive when valence drops below neutral
    # Higher arousal from neutral = more activated = more stressed
    delta_a = arousal - neutral_a  # positive when arousal rises above neutral

    # Weighted combination
    stress_score = np.clip((delta_v * 0.5 + delta_a * 0.5), 0.0, 1.0)

    return float(stress_score)


def compute_feature_zscore(current_features, baseline, feature_type='au'):
    """
    Compute Z-score deviation of current features from baseline.

    Args:
        current_features: current AU or audio feature vector
        baseline: loaded baseline dict
        feature_type: 'au' or 'audio'

    Returns:
        z_scores: array of per-feature Z-scores
        mean_zscore: average absolute Z-score
    """
    mean_key = f'{feature_type}_mean'
    std_key  = f'{feature_type}_std'

    if mean_key not in baseline or std_key not in baseline:
        return np.zeros_like(current_features), 0.0

    mean = np.array(baseline[mean_key])
    std  = np.array(baseline[std_key])

    # Avoid division by zero
    std = np.where(std < 1e-6, 1.0, std)

    z_scores = (current_features - mean) / std
    mean_zscore = float(np.mean(np.abs(z_scores)))

    return z_scores, mean_zscore


def classify_va_zone(valence, arousal):
    """
    Classify the V-A point into an emotional zone using
    non-overlapping quadrant-based thresholds.

    V-A space mapping:
      High Arousal + Negative Valence → stress / anger / anxiety
      High Arousal + Positive Valence → joy / excitement
      Low Arousal  + Negative Valence → sadness
      Low Arousal  + Positive Valence → calm / relaxed

    Returns:
        str: zone name
    """
    # Negative valence, high arousal quadrant
    if valence < -0.3 and arousal > 0.4:
        return 'high_stress'
    elif valence < -0.3 and 0.1 < arousal <= 0.4:
        return 'anger'
    elif -0.3 <= valence < 0.0 and arousal > 0.3:
        return 'anxiety'

    # Negative valence, low arousal quadrant
    elif valence < -0.1 and arousal <= 0.1:
        return 'sadness'

    # Positive valence, high arousal quadrant
    elif valence > 0.2 and arousal > 0.2:
        return 'joy'

    # Positive valence, low arousal quadrant
    elif valence > 0.0 and arousal <= 0.2:
        return 'calm'

    else:
        return 'neutral'


def get_recommendations(zone):
    """
    Get personalized stress-management recommendations based on V-A zone.

    Returns:
        list of recommendation strings
    """
    recommendations = {
        'high_stress': [
            'Take a 5-minute break from your current task',
            'Practice box breathing (4-4-4-4): inhale 4s, hold 4s, exhale 4s, hold 4s',
            'Try a guided mindfulness meditation (3-5 minutes)',
            'Disconnect from screen and walk in nature',
            'Reach out to someone you trust for social support',
        ],
        'anxiety': [
            'Grounding exercise: name 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste',
            'Try a guided meditation focused on anxiety relief',
            'Write down your current thoughts and worries',
            'Practice visualization: imagine your safe place',
            'Establish a predictable routine for the next hour',
        ],
        'anger': [
            'Step away from the current trigger immediately',
            'Engage in brief physical exercise (pushups, jumping jacks)',
            'Practice deep abdominal breathing for 2 minutes',
            'Journal your feelings without censoring',
            'Apply cold water on face or wrists to activate the dive reflex',
        ],
        'sadness': [
            'Reach out to a trusted friend or family member',
            'Engage in a comfort activity you enjoy',
            'Light physical movement: a short walk or stretching',
            'Reflect on positive memories or achievements',
            'Consider seeking professional support if this persists',
        ],
        'joy': [
            'Maintain your current positive routine',
            'Share your positive energy with others around you',
            'Note what caused this positive state for future reference',
            'Use this productive time for creative work',
            'Practice gratitude journaling',
        ],
        'calm': [
            'Stay hydrated — drink some water',
            'Maintain your balance with regular breaks',
            'Practice mindfulness to sustain this calm state',
            'Check in with your emotions periodically',
            'This is a great time for focused, deep work',
        ],
        'neutral': [
            'Take regular short breaks (5 min every 30 min)',
            'Stretch every 30 minutes to maintain energy',
            'Stay hydrated throughout the session',
            'Consider mild physical activity to boost mood',
            'Practice brief breathing exercises to stay centered',
        ],
    }

    return recommendations.get(zone, recommendations['neutral'])


# ──────────────────────────────────────────────────────────────
#  Demo / Test
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Calibration Module — Demo")
    print("=" * 60)

    # Simulate 90 seconds of calibration data
    print("\n📊 Simulating 90-second calibration...")
    n_au_samples = 90 * 30    # 30 fps for 90 seconds
    n_audio_samples = 90 * 2  # 2 chunks per second (500ms each)

    au_vectors = [np.random.randn(17).astype(np.float32) * 0.5 for _ in range(n_au_samples)]
    audio_features = [np.random.randn(84).astype(np.float32) for _ in range(n_audio_samples)]

    # Compute baseline
    baseline = compute_baseline_from_data(au_vectors, audio_features)
    print(f"Baseline computed with {baseline['num_au_samples']} AU samples "
          f"and {baseline['num_audio_samples']} audio samples")

    # Save baseline
    test_user = 'demo_user'
    save_baseline(test_user, baseline)

    # Load and verify
    loaded = load_baseline(test_user)
    print(f"Loaded baseline: calibrated at {loaded['calibrated_at']}")

    # Test stress computation
    va_stressed = [-0.5, 0.7]  # Negative valence, high arousal
    va_calm = [0.3, -0.1]     # Positive valence, low arousal

    stress_high = compute_personalized_stress(va_stressed, test_user)
    stress_low  = compute_personalized_stress(va_calm, test_user)

    print(f"\nStress score (stressed state): {stress_high:.3f}")
    print(f"Stress score (calm state):     {stress_low:.3f}")

    # Test zone classification
    zone_stressed = classify_va_zone(-0.5, 0.7)
    zone_calm     = classify_va_zone(0.3, -0.1)
    print(f"\nZone (stressed): {zone_stressed}")
    print(f"Zone (calm):     {zone_calm}")

    # Test recommendations
    recs = get_recommendations(zone_stressed)
    print(f"\nRecommendations for '{zone_stressed}':")
    for r in recs:
        print(f"  • {r}")

    print("\n✅ Calibration module test complete!")
