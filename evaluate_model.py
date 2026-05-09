"""
evaluate_model.py — Train & Evaluate Stress Detection Model Accuracy
Generates: full_model.h5 + accuracy_report.png + detailed console metrics
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model_training import (
    build_full_model, generate_synthetic_data,
    concordance_cc, ccc_loss
)

def evaluate():
    print("=" * 70)
    print("  🧠 Stress Detection Model — Training & Accuracy Evaluation")
    print("=" * 70)

    # ── 1. Generate Data ─────────────────────────────────────────
    au_data, audio_data, labels = generate_synthetic_data(n_samples=5000)

    # Train / Val / Test split (60/20/20)
    n = len(labels)
    idx = np.random.permutation(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    au_train, audio_train, y_train = au_data[train_idx], audio_data[train_idx], labels[train_idx]
    au_val, audio_val, y_val = au_data[val_idx], audio_data[val_idx], labels[val_idx]
    au_test, audio_test, y_test = au_data[test_idx], audio_data[test_idx], labels[test_idx]

    print(f"\n📊 Dataset Split:")
    print(f"   Train : {len(y_train)} samples")
    print(f"   Val   : {len(y_val)} samples")
    print(f"   Test  : {len(y_test)} samples")

    # ── 2. Build & Compile Model ─────────────────────────────────
    model = build_full_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )

    print(f"\n🏗️  Model Parameters: {model.count_params():,}")

    # ── 3. Train ─────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
    ]

    print("\n🚀 Training started...")
    history = model.fit(
        [au_train, audio_train], y_train,
        validation_data=([au_val, audio_val], y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # ── 4. Predict on Test Set ───────────────────────────────────
    print("\n📏 Evaluating on held-out TEST set...")
    y_pred = model.predict([au_test, audio_test], verbose=0)

    v_true, a_true = y_test[:, 0], y_test[:, 1]
    v_pred, a_pred = y_pred[:, 0], y_pred[:, 1]

    # ── 5. Compute Metrics ───────────────────────────────────────
    # CCC
    ccc_v = float(concordance_cc(
        tf.constant(v_true, dtype=tf.float32),
        tf.constant(v_pred, dtype=tf.float32)
    ).numpy())
    ccc_a = float(concordance_cc(
        tf.constant(a_true, dtype=tf.float32),
        tf.constant(a_pred, dtype=tf.float32)
    ).numpy())

    # MAE
    mae_v = mean_absolute_error(v_true, v_pred)
    mae_a = mean_absolute_error(a_true, a_pred)

    # MSE
    mse_v = mean_squared_error(v_true, v_pred)
    mse_a = mean_squared_error(a_true, a_pred)

    # RMSE
    rmse_v = np.sqrt(mse_v)
    rmse_a = np.sqrt(mse_a)

    # R²
    r2_v = r2_score(v_true, v_pred)
    r2_a = r2_score(a_true, a_pred)

    # Pearson Correlation
    pearson_v = float(np.corrcoef(v_true, v_pred)[0, 1])
    pearson_a = float(np.corrcoef(a_true, a_pred)[0, 1])

    print("\n" + "=" * 70)
    print("  📈 MODEL ACCURACY REPORT (Test Set)")
    print("=" * 70)
    print(f"{'Metric':<30} {'Valence':>12} {'Arousal':>12}")
    print("-" * 55)
    print(f"{'CCC (Concordance)':<30} {ccc_v:>12.4f} {ccc_a:>12.4f}")
    print(f"{'Pearson Correlation':<30} {pearson_v:>12.4f} {pearson_a:>12.4f}")
    print(f"{'R² Score':<30} {r2_v:>12.4f} {r2_a:>12.4f}")
    print(f"{'MAE (Mean Abs Error)':<30} {mae_v:>12.4f} {mae_a:>12.4f}")
    print(f"{'MSE (Mean Sq Error)':<30} {mse_v:>12.4f} {mse_a:>12.4f}")
    print(f"{'RMSE (Root MSE)':<30} {rmse_v:>12.4f} {rmse_a:>12.4f}")
    print("-" * 55)

    # Overall Quality Rating
    avg_ccc = (ccc_v + ccc_a) / 2
    if avg_ccc >= 0.70:
        quality = "🟢 EXCELLENT"
    elif avg_ccc >= 0.50:
        quality = "🟡 GOOD"
    elif avg_ccc >= 0.30:
        quality = "🟠 MODERATE"
    elif avg_ccc >= 0.10:
        quality = "🔴 POOR"
    else:
        quality = "⚫ VERY POOR"

    print(f"\n{'Overall Quality':<30} {quality}")
    print(f"{'Average CCC':<30} {avg_ccc:>12.4f}")

    # ── 6. Generate Accuracy Report Plot ─────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Model Accuracy Report', fontsize=22, fontweight='bold',
                 color='#e2e8f0', y=0.98)

    # --- Plot 1: Training Loss Curves ---
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor('#16213e')
    ax1.plot(history.history['loss'], label='Train Loss', color='#6366f1', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', color='#f43f5e', linewidth=2)
    ax1.set_title('Training & Validation Loss', color='#e2e8f0', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', color='#e2e8f0')
    ax1.set_ylabel('MSE Loss', color='#e2e8f0')
    ax1.legend(framealpha=0.7)
    ax1.grid(True, alpha=0.2, color='#334155')
    ax1.tick_params(colors='#e2e8f0')

    # --- Plot 2: MAE Curves ---
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor('#16213e')
    ax2.plot(history.history['mae'], label='Train MAE', color='#22c55e', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Val MAE', color='#fbbf24', linewidth=2)
    ax2.set_title('Mean Absolute Error', color='#e2e8f0', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', color='#e2e8f0')
    ax2.set_ylabel('MAE', color='#e2e8f0')
    ax2.legend(framealpha=0.7)
    ax2.grid(True, alpha=0.2, color='#334155')
    ax2.tick_params(colors='#e2e8f0')

    # --- Plot 3: CCC Bar Chart ---
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor('#16213e')
    bars = ax3.bar(['Valence\nCCC', 'Arousal\nCCC', 'Average\nCCC'],
                   [ccc_v, ccc_a, avg_ccc],
                   color=['#6366f1', '#f43f5e', '#22c55e'],
                   edgecolor='white', linewidth=1.5, width=0.5)
    for bar, val in zip(bars, [ccc_v, ccc_a, avg_ccc]):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', color='#e2e8f0',
                fontweight='bold', fontsize=12)
    ax3.axhline(y=0.7, color='#22c55e', linestyle='--', linewidth=1.5, label='Target ≥ 0.70')
    ax3.axhline(y=0.5, color='#fbbf24', linestyle='--', linewidth=1, label='Acceptable ≥ 0.50')
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_title('Concordance Correlation (CCC)', color='#e2e8f0', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, framealpha=0.7)
    ax3.grid(True, alpha=0.2, color='#334155', axis='y')
    ax3.tick_params(colors='#e2e8f0')

    # --- Plot 4: Valence Scatter (True vs Predicted) ---
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor('#16213e')
    ax4.scatter(v_true, v_pred, alpha=0.3, s=15, c='#6366f1', edgecolors='none')
    ax4.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect Prediction')
    ax4.set_xlim(-1.1, 1.1)
    ax4.set_ylim(-1.1, 1.1)
    ax4.set_xlabel('True Valence', color='#e2e8f0')
    ax4.set_ylabel('Predicted Valence', color='#e2e8f0')
    ax4.set_title(f'Valence: True vs Predicted (R²={r2_v:.3f})', color='#e2e8f0',
                  fontsize=13, fontweight='bold')
    ax4.legend(framealpha=0.7)
    ax4.grid(True, alpha=0.2, color='#334155')
    ax4.tick_params(colors='#e2e8f0')
    ax4.set_aspect('equal')

    # --- Plot 5: Arousal Scatter (True vs Predicted) ---
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor('#16213e')
    ax5.scatter(a_true, a_pred, alpha=0.3, s=15, c='#f43f5e', edgecolors='none')
    ax5.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect Prediction')
    ax5.set_xlim(-1.1, 1.1)
    ax5.set_ylim(-1.1, 1.1)
    ax5.set_xlabel('True Arousal', color='#e2e8f0')
    ax5.set_ylabel('Predicted Arousal', color='#e2e8f0')
    ax5.set_title(f'Arousal: True vs Predicted (R²={r2_a:.3f})', color='#e2e8f0',
                  fontsize=13, fontweight='bold')
    ax5.legend(framealpha=0.7)
    ax5.grid(True, alpha=0.2, color='#334155')
    ax5.tick_params(colors='#e2e8f0')
    ax5.set_aspect('equal')

    # --- Plot 6: Error Distribution ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor('#16213e')
    errors_v = v_pred - v_true
    errors_a = a_pred - a_true
    ax6.hist(errors_v, bins=40, alpha=0.6, color='#6366f1', label=f'Valence (μ={np.mean(errors_v):.3f})', edgecolor='white', linewidth=0.5)
    ax6.hist(errors_a, bins=40, alpha=0.6, color='#f43f5e', label=f'Arousal (μ={np.mean(errors_a):.3f})', edgecolor='white', linewidth=0.5)
    ax6.axvline(x=0, color='#fbbf24', linestyle='--', linewidth=2)
    ax6.set_xlabel('Prediction Error', color='#e2e8f0')
    ax6.set_ylabel('Count', color='#e2e8f0')
    ax6.set_title('Error Distribution', color='#e2e8f0', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9, framealpha=0.7)
    ax6.grid(True, alpha=0.2, color='#334155')
    ax6.tick_params(colors='#e2e8f0')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    report_path = os.path.join(os.path.dirname(__file__), 'accuracy_report.png')
    fig.savefig(report_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"\n📊 Accuracy report saved to: {report_path}")

    # ── 7. Save Model ───────────────────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), 'full_model.h5')
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")

    print("\n✅ Evaluation complete!")
    return {
        'ccc_valence': ccc_v,
        'ccc_arousal': ccc_a,
        'mae_valence': mae_v,
        'mae_arousal': mae_a,
        'r2_valence': r2_v,
        'r2_arousal': r2_a,
        'quality': quality,
    }


if __name__ == '__main__':
    evaluate()
