"""
model_training.py — Full end-to-end model training script
Builds: Facial LSTM + Speech LSTM + Cross-Modal Attention + VA Regression Head
Trains on synthetic data (replace with real datasets: Ulm-TSST, StressID, MAHNOB-HCI)
Saves: full_model.h5
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Conv1D, MaxPooling1D, Dropout,
    Concatenate, LayerNormalization, MultiHeadAttention,
    Reshape, Add, Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ──────────────────────────────────────────────────────────────
#  Custom Metric: Concordance Correlation Coefficient (CCC)
# ──────────────────────────────────────────────────────────────
def concordance_cc(y_true, y_pred):
    """Concordance Correlation Coefficient — standard AVEC metric."""
    mean_t = tf.reduce_mean(y_true)
    mean_p = tf.reduce_mean(y_pred)
    var_t  = tf.math.reduce_variance(y_true)
    var_p  = tf.math.reduce_variance(y_pred)
    cov    = tf.reduce_mean((y_true - mean_t) * (y_pred - mean_p))
    return (2.0 * cov) / (var_t + var_p + tf.square(mean_t - mean_p) + 1e-8)

def ccc_loss(y_true, y_pred):
    """1 - CCC as a loss function."""
    return 1.0 - concordance_cc(y_true, y_pred)

def combined_ccc_mse_loss(y_true, y_pred):
    """Combined CCC + MSE loss for stable, accurate V-A regression."""
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ccc_v = concordance_cc(y_true[:, 0], y_pred[:, 0])
    ccc_a = concordance_cc(y_true[:, 1], y_pred[:, 1])
    ccc_term = 1.0 - (ccc_v + ccc_a) / 2.0
    return 0.4 * mse + 0.6 * ccc_term

# ──────────────────────────────────────────────────────────────
#  Sub-model 1: Facial Temporal Model (Bidirectional LSTM)
# ──────────────────────────────────────────────────────────────
def build_facial_temporal_model(seq_len=30, au_dim=17, embed_dim=128):
    """
    Processes 30-frame AU intensity sequences (1 second @ 30fps).
    Input shape : (batch, 30, 17)
    Output shape: (batch, 128)
    """
    from tensorflow.keras.layers import Bidirectional, BatchNormalization

    inp = Input(shape=(seq_len, au_dim), name='au_input')
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
    x = Dense(embed_dim, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(embed_dim, activation='relu')(x)
    return Model(inp, out, name='facial_temporal')

# ──────────────────────────────────────────────────────────────
#  Sub-model 2: Speech Temporal Model (1D-CNN + LSTM)
# ──────────────────────────────────────────────────────────────
def build_speech_temporal_model(seq_len=20, feat_dim=84, embed_dim=128):
    """
    Processes 20 × 500ms acoustic feature chunks = 10 seconds of audio.
    Input shape : (batch, 20, 84)
    Output shape: (batch, 128)
    """
    from tensorflow.keras.layers import Bidirectional, BatchNormalization

    inp = Input(shape=(seq_len, feat_dim), name='audio_input')
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
    x = Dense(embed_dim, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(embed_dim, activation='relu')(x)
    return Model(inp, out, name='speech_temporal')

# ──────────────────────────────────────────────────────────────
#  Sub-model 3: Cross-Modal Attention Fusion
# ──────────────────────────────────────────────────────────────
def build_fusion_model(embed_dim=128):
    """
    Cross-Modal Attention: face attends to speech & vice-versa.
    Input : two (batch, 128) vectors
    Output: (batch, 256) fused vector
    """
    face_inp   = Input(shape=(embed_dim,), name='face_features')
    speech_inp = Input(shape=(embed_dim,), name='speech_features')

    face_seq   = Reshape((1, embed_dim))(face_inp)      # (B,1,128)
    speech_seq = Reshape((1, embed_dim))(speech_inp)     # (B,1,128)

    # Face attends to speech
    face_attn   = MultiHeadAttention(num_heads=4, key_dim=32)(face_seq, speech_seq)
    # Speech attends to face
    speech_attn = MultiHeadAttention(num_heads=4, key_dim=32)(speech_seq, face_seq)

    face_attn_flat   = Flatten()(face_attn)     # (B, 128)
    speech_attn_flat = Flatten()(speech_attn)    # (B, 128)

    face_out   = LayerNormalization()(Add()([face_attn_flat, face_inp]))
    speech_out = LayerNormalization()(Add()([speech_attn_flat, speech_inp]))

    fused = Concatenate()([face_out, speech_out])  # (B, 256)
    fused = Dense(256, activation='relu')(fused)
    fused = Dropout(0.3)(fused)
    fused = Dense(128, activation='relu')(fused)
    fused = Dropout(0.2)(fused)
    return Model([face_inp, speech_inp], fused, name='cross_modal_fusion')

# ──────────────────────────────────────────────────────────────
#  Sub-model 4: Valence-Arousal Regression Head
# ──────────────────────────────────────────────────────────────
def build_va_regression_head(input_dim=128):
    """
    Two output neurons: Valence ∈ [-1,+1], Arousal ∈ [-1,+1].
    """
    inp    = Input(shape=(input_dim,), name='fused_input')
    x      = Dense(64, activation='relu')(inp)
    x      = Dropout(0.15)(x)
    x      = Dense(32, activation='relu')(x)
    va_out = Dense(2, activation='tanh', name='valence_arousal')(x)
    return Model(inp, va_out, name='va_head')

# ──────────────────────────────────────────────────────────────
#  Full End-to-End Model
# ──────────────────────────────────────────────────────────────
def build_full_model(au_seq_len=30, au_dim=17, audio_seq_len=20, audio_feat_dim=84, embed_dim=128):
    """Compose all four sub-models into one trainable graph."""
    # Inputs
    au_input    = Input(shape=(au_seq_len, au_dim), name='au_input')
    audio_input = Input(shape=(audio_seq_len, audio_feat_dim), name='audio_input')

    # Sub-models
    facial_model = build_facial_temporal_model(au_seq_len, au_dim, embed_dim)
    speech_model = build_speech_temporal_model(audio_seq_len, audio_feat_dim, embed_dim)
    fusion_model = build_fusion_model(embed_dim)
    va_head      = build_va_regression_head(128)

    # Forward pass
    face_embed  = facial_model(au_input)
    speech_embed = speech_model(audio_input)
    fused       = fusion_model([face_embed, speech_embed])
    va_output   = va_head(fused)

    model = Model(inputs=[au_input, audio_input], outputs=va_output, name='stress_detection_model')
    return model

# ──────────────────────────────────────────────────────────────
#  Synthetic Data Generator — Label-Driven Feature Construction
# ──────────────────────────────────────────────────────────────

# Research-based AU-emotion mapping coefficients
# Positive = AU increases when label component is positive
_AU_VALENCE_MAP = {
    0:  0.3,   # AU01 Inner Brow Raise → surprise (moderate +V)
    1:  0.2,   # AU02 Outer Brow Raise → surprise
    2: -0.8,   # AU04 Brow Furrow → negative valence (stress)
    3:  0.1,   # AU05 Upper Lid Raise → surprise
    4:  0.6,   # AU06 Cheek Raise → positive (Duchenne smile)
    5: -0.7,   # AU07 Eyelid Tightener → stress/tension
    6: -0.5,   # AU09 Nose Wrinkle → disgust/negative
    7: -0.3,   # AU10 Upper Lip Raiser → negative
    8:  0.9,   # AU12 Lip Corner Pull → smile → positive valence
    9:  0.1,   # AU14 Dimpler
    10:-0.6,   # AU15 Lip Corner Depressor → negative (frown)
    11:-0.4,   # AU17 Chin Raise → negative
    12:-0.3,   # AU20 Lip Stretcher → fear/stress
    13:-0.5,   # AU23 Lip Tightener → tension/stress
    14: 0.2,   # AU25 Lips Part → neutral
    15: 0.1,   # AU26 Jaw Drop → surprise
    16: 0.0,   # AU45 Blink → neutral
}

_AU_AROUSAL_MAP = {
    0:  0.4,   # AU01 → surprise → high arousal
    1:  0.3,   # AU02 → surprise → high arousal
    2:  0.6,   # AU04 → stress → high arousal
    3:  0.5,   # AU05 → wide eyes → high arousal
    4:  0.3,   # AU06 → engagement
    5:  0.7,   # AU07 → tension → high arousal
    6:  0.4,   # AU09 → disgust activation
    7:  0.3,   # AU10 → activation
    8:  0.2,   # AU12 → smile → moderate arousal
    9:  0.1,   # AU14
    10: 0.1,   # AU15 → low arousal sadness variant
    11: 0.3,   # AU17 → tension
    12: 0.5,   # AU20 → fear → high arousal
    13: 0.4,   # AU23 → tension → arousal
    14: 0.3,   # AU25 → speaking/activation
    15: 0.4,   # AU26 → surprise → arousal
    16: -0.2,  # AU45 → blink → low arousal
}


def generate_synthetic_data(n_samples=15000, au_seq_len=30, au_dim=17,
                            audio_seq_len=20, audio_feat_dim=84):
    """
    Generate synthetic multimodal stress data with STRONG feature-label
    correlations using label-driven feature construction.

    Strategy: Generate V-A labels FIRST, then construct features that
    are correlated with those labels. This ensures the model can learn
    meaningful feature→label mappings.
    """
    print("📊 Generating label-driven synthetic training data...")
    np.random.seed(42)

    # ── Step 1: Generate balanced V-A labels covering all zones ──
    # Generate labels that cover the full V-A space with good zone coverage
    n_per_zone = n_samples // 7
    n_leftover = n_samples - n_per_zone * 7

    zone_params = {
        # zone: (valence_center, arousal_center, v_spread, a_spread)
        'high_stress': (-0.6,  0.65, 0.15, 0.15),
        'anger':       (-0.55, 0.25, 0.15, 0.10),
        'anxiety':     (-0.15, 0.50, 0.10, 0.12),
        'sadness':     (-0.45,-0.20, 0.20, 0.15),
        'joy':         ( 0.55, 0.50, 0.20, 0.15),
        'calm':        ( 0.35,-0.05, 0.20, 0.15),
        'neutral':     ( 0.0,  0.10, 0.15, 0.12),
    }

    valence_parts = []
    arousal_parts = []
    for zone_name, (v_c, a_c, v_s, a_s) in zone_params.items():
        n_z = n_per_zone + (n_leftover if zone_name == 'neutral' else 0)
        v = np.clip(np.random.normal(v_c, v_s, n_z), -1.0, 1.0).astype(np.float32)
        a = np.clip(np.random.normal(a_c, a_s, n_z), -1.0, 1.0).astype(np.float32)
        valence_parts.append(v)
        arousal_parts.append(a)

    valence = np.concatenate(valence_parts)
    arousal = np.concatenate(arousal_parts)

    # Shuffle to mix zones together
    shuffle_idx = np.random.permutation(len(valence))
    valence = valence[shuffle_idx]
    arousal = arousal[shuffle_idx]
    n_samples = len(valence)  # Adjust in case of rounding

    labels = np.stack([valence, arousal], axis=1)  # (n_samples, 2)

    # ── Step 2: Construct AU features from V-A labels ──
    au_data = np.zeros((n_samples, au_seq_len, au_dim), dtype=np.float32)

    for i in range(n_samples):
        v, a = valence[i], arousal[i]

        # Base AU vector derived from V-A via research mappings
        au_base = np.zeros(au_dim, dtype=np.float32)
        for j in range(au_dim):
            v_coef = _AU_VALENCE_MAP.get(j, 0.0)
            a_coef = _AU_AROUSAL_MAP.get(j, 0.0)
            # AU intensity = weighted combination of valence&arousal influence
            au_base[j] = np.clip(
                1.5 + v * v_coef * 2.0 + a * a_coef * 2.0,
                0.0, 5.0
            )

        # Create temporal sequence with small frame-to-frame variation
        for t in range(au_seq_len):
            temporal_noise = np.random.normal(0, 0.15, au_dim).astype(np.float32)
            au_data[i, t, :] = np.clip(au_base + temporal_noise, 0.0, 5.0)

    # ── Step 3: Construct audio features from V-A labels ──
    audio_data = np.zeros((n_samples, audio_seq_len, audio_feat_dim), dtype=np.float32)

    for i in range(n_samples):
        v, a = valence[i], arousal[i]

        # Key acoustic stress indicators (first few features)
        base_pitch   =  180.0 + a * 60.0 - v * 20.0   # Stress: higher pitch
        base_jitter  =  0.02 + max(0, -v) * 0.03 + max(0, a) * 0.02  # Stress: more jitter
        base_shimmer =  0.04 + max(0, -v) * 0.04 + max(0, a) * 0.03
        base_rms     =  0.05 + a * 0.03   # Arousal: louder
        base_hnr     = 15.0 + v * 5.0 - max(0, a) * 3.0  # Stress: lower HNR
        base_rate    =  4.0 + a * 1.5     # Arousal: faster speech rate

        for t in range(audio_seq_len):
            feat = np.random.normal(0, 0.08, audio_feat_dim).astype(np.float32)
            # Set key features with strong correlation
            feat[0]  = base_pitch   + np.random.normal(0, 8.0)
            feat[1]  = base_jitter  + np.random.normal(0, 0.005)
            feat[2]  = base_shimmer + np.random.normal(0, 0.008)
            feat[3]  = base_rms     + np.random.normal(0, 0.01)
            feat[4]  = base_hnr     + np.random.normal(0, 1.5)
            feat[5]  = base_rate    + np.random.normal(0, 0.3)
            # MFCC-like features (6-18): correlated with emotional state
            for m in range(6, min(19, audio_feat_dim)):
                feat[m] = v * (0.3 - 0.02 * m) + a * (0.2 - 0.01 * m) + np.random.normal(0, 0.15)
            # Spectral features (19-40): correlated with arousal
            for s in range(19, min(41, audio_feat_dim)):
                feat[s] = a * 0.15 + np.random.normal(0, 0.1)
            audio_data[i, t, :] = feat

    print(f"   Generated {n_samples} samples with {au_seq_len}×{au_dim} AU + {audio_seq_len}×{audio_feat_dim} audio")
    print(f"   V range: [{valence.min():.2f}, {valence.max():.2f}]  A range: [{arousal.min():.2f}, {arousal.max():.2f}]")

    return au_data, audio_data, labels

# ──────────────────────────────────────────────────────────────
#  Training Script
# ──────────────────────────────────────────────────────────────
def train_model():
    print("=" * 70)
    print("  Real-Time Stress Detection — Model Training (v2)")
    print("  Bidirectional LSTM + CNN-LSTM + Cross-Modal Attention + CCC Loss")
    print("=" * 70)

    # 1. Generate data
    au_data, audio_data, labels = generate_synthetic_data(n_samples=15000)

    # 2. Shuffled train/val split
    n = len(labels)
    np.random.seed(123)
    idx = np.random.permutation(n)
    split = int(0.85 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    au_train, au_val         = au_data[train_idx], au_data[val_idx]
    audio_train, audio_val   = audio_data[train_idx], audio_data[val_idx]
    labels_train, labels_val = labels[train_idx], labels[val_idx]

    print(f"\n🔢 Training samples  : {len(train_idx)}")
    print(f"🔢 Validation samples: {len(val_idx)}")

    # 3. Build model
    model = build_full_model()
    model.summary()

    # 4. Compile with combined CCC+MSE loss
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss=combined_ccc_mse_loss,
        metrics=['mae']
    )

    # 5. Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1),
        ModelCheckpoint('full_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
    ]

    # 6. Train
    print("\n🚀 Starting training...")
    history = model.fit(
        [au_train, audio_train], labels_train,
        validation_data=([au_val, audio_val], labels_val),
        epochs=80,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Evaluate CCC
    val_pred = model.predict([au_val, audio_val])
    ccc_v = concordance_cc(
        tf.constant(labels_val[:, 0], dtype=tf.float32),
        tf.constant(val_pred[:, 0], dtype=tf.float32)
    ).numpy()
    ccc_a = concordance_cc(
        tf.constant(labels_val[:, 1], dtype=tf.float32),
        tf.constant(val_pred[:, 1], dtype=tf.float32)
    ).numpy()
    print(f"\n📈 Validation CCC (Valence) : {ccc_v:.4f}")
    print(f"📈 Validation CCC (Arousal) : {ccc_a:.4f}")
    print(f"📈 Average CCC              : {(ccc_v + ccc_a)/2:.4f}")

    # 8. Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Combined CCC+MSE Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # CCC bar chart
    axes[2].bar(['Valence CCC', 'Arousal CCC'], [ccc_v, ccc_a],
                color=['#6366f1', '#f43f5e'], edgecolor='white', linewidth=2)
    axes[2].set_ylim([-1, 1])
    axes[2].axhline(y=0.7, color='green', linestyle='--', label='Target ≥ 0.70')
    axes[2].set_title('Concordance Correlation', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Training curves saved to training_curves.png")

    # 9. Save final model in both formats
    model.save('full_model.h5')
    model.save('full_model.keras')
    print("💾 Model saved to full_model.h5 and full_model.keras")
    print("\n✅ Training complete!")

    return model, history

if __name__ == '__main__':
    train_model()
