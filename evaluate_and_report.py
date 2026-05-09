"""
evaluate_and_report.py — Train, evaluate, and report model accuracy.
Saves model in native Keras format for compatibility.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model_training import (
    build_full_model, generate_synthetic_data, concordance_cc
)

def run():
    print("=" * 60)
    print("  Model Accuracy Evaluation")
    print("=" * 60)

    # 1. Data
    au_data, audio_data, labels = generate_synthetic_data(n_samples=5000)
    n = len(labels)
    np.random.seed(42)
    idx = np.random.permutation(n)
    train_end, val_end = int(0.6*n), int(0.8*n)

    au_train, audio_train, y_train = au_data[idx[:train_end]], audio_data[idx[:train_end]], labels[idx[:train_end]]
    au_val, audio_val, y_val = au_data[idx[train_end:val_end]], audio_data[idx[train_end:val_end]], labels[idx[train_end:val_end]]
    au_test, audio_test, y_test = au_data[idx[val_end:]], audio_data[idx[val_end:]], labels[idx[val_end:]]

    print(f'Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}')

    # 2. Build & Train
    model = build_full_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse', metrics=['mae'])
    print(f'Parameters: {model.count_params():,}')

    history = model.fit(
        [au_train, audio_train], y_train,
        validation_data=([au_val, audio_val], y_val),
        epochs=50, batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ],
        verbose=1
    )

    # 3. Predict on test
    y_pred = model.predict([au_test, audio_test], verbose=0)
    v_true, a_true = y_test[:, 0], y_test[:, 1]
    v_pred, a_pred = y_pred[:, 0], y_pred[:, 1]

    # 4. Metrics
    ccc_v = float(concordance_cc(tf.constant(v_true, tf.float32), tf.constant(v_pred, tf.float32)).numpy())
    ccc_a = float(concordance_cc(tf.constant(a_true, tf.float32), tf.constant(a_pred, tf.float32)).numpy())
    mae_v = mean_absolute_error(v_true, v_pred)
    mae_a = mean_absolute_error(a_true, a_pred)
    mse_v = mean_squared_error(v_true, v_pred)
    mse_a = mean_squared_error(a_true, a_pred)
    r2_v = r2_score(v_true, v_pred)
    r2_a = r2_score(a_true, a_pred)
    pearson_v = float(np.corrcoef(v_true, v_pred)[0, 1])
    pearson_a = float(np.corrcoef(a_true, a_pred)[0, 1])
    avg_ccc = (ccc_v + ccc_a) / 2

    print()
    print('=' * 60)
    print('  MODEL ACCURACY REPORT - Test:', len(y_test), 'samples')
    print('=' * 60)
    print('{:<30} {:>12} {:>12}'.format('Metric', 'Valence', 'Arousal'))
    print('-' * 55)
    print('{:<30} {:>12.4f} {:>12.4f}'.format('CCC', ccc_v, ccc_a))
    print('{:<30} {:>12.4f} {:>12.4f}'.format('Pearson Correlation', pearson_v, pearson_a))
    print('{:<30} {:>12.4f} {:>12.4f}'.format('R2 Score', r2_v, r2_a))
    print('{:<30} {:>12.4f} {:>12.4f}'.format('MAE', mae_v, mae_a))
    print('{:<30} {:>12.4f} {:>12.4f}'.format('MSE', mse_v, mse_a))
    print('{:<30} {:>12.4f} {:>12.4f}'.format('RMSE', np.sqrt(mse_v), np.sqrt(mse_a)))
    print('-' * 55)
    print('{:<30} {:>12.4f}'.format('Average CCC', avg_ccc))

    if avg_ccc >= 0.70: q = 'EXCELLENT'
    elif avg_ccc >= 0.50: q = 'GOOD'
    elif avg_ccc >= 0.30: q = 'MODERATE'
    elif avg_ccc >= 0.10: q = 'POOR'
    else: q = 'VERY POOR'
    print('{:<30} {:>12}'.format('Quality Rating', q))
    print('=' * 60)

    # 5. Save plots
    fig = plt.figure(figsize=(20, 14), facecolor='#1a1a2e')
    fig.suptitle('Model Accuracy Report', fontsize=22, fontweight='bold', color='#e2e8f0', y=0.98)

    # Loss
    ax1 = fig.add_subplot(2, 3, 1, facecolor='#16213e')
    ax1.plot(history.history['loss'], label='Train', color='#6366f1', lw=2)
    ax1.plot(history.history['val_loss'], label='Val', color='#f43f5e', lw=2)
    ax1.set_title('MSE Loss', color='#e2e8f0', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.2, color='#334155')
    ax1.tick_params(colors='#e2e8f0'); ax1.set_xlabel('Epoch', color='#e2e8f0')

    # MAE
    ax2 = fig.add_subplot(2, 3, 2, facecolor='#16213e')
    ax2.plot(history.history['mae'], label='Train', color='#22c55e', lw=2)
    ax2.plot(history.history['val_mae'], label='Val', color='#fbbf24', lw=2)
    ax2.set_title('MAE', color='#e2e8f0', fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.2, color='#334155')
    ax2.tick_params(colors='#e2e8f0'); ax2.set_xlabel('Epoch', color='#e2e8f0')

    # CCC bars
    ax3 = fig.add_subplot(2, 3, 3, facecolor='#16213e')
    bars = ax3.bar(['Valence', 'Arousal', 'Average'], [ccc_v, ccc_a, avg_ccc],
                   color=['#6366f1','#f43f5e','#22c55e'], edgecolor='white', width=0.5)
    for b, v in zip(bars, [ccc_v, ccc_a, avg_ccc]):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.3f}',
                ha='center', color='#e2e8f0', fontweight='bold', fontsize=12)
    ax3.axhline(y=0.7, color='#22c55e', ls='--', lw=1.5, label='Target 0.70')
    ax3.set_ylim(-0.1, 1.1); ax3.set_title('CCC', color='#e2e8f0', fontweight='bold')
    ax3.legend(fontsize=9); ax3.tick_params(colors='#e2e8f0')

    # Valence scatter
    ax4 = fig.add_subplot(2, 3, 4, facecolor='#16213e')
    ax4.scatter(v_true, v_pred, alpha=0.3, s=15, c='#6366f1')
    ax4.plot([-1,1],[-1,1],'r--', lw=2)
    ax4.set_title(f'Valence R2={r2_v:.3f}', color='#e2e8f0', fontweight='bold')
    ax4.set_xlim(-1.1,1.1); ax4.set_ylim(-1.1,1.1); ax4.set_aspect('equal')
    ax4.tick_params(colors='#e2e8f0'); ax4.grid(True, alpha=0.2, color='#334155')

    # Arousal scatter
    ax5 = fig.add_subplot(2, 3, 5, facecolor='#16213e')
    ax5.scatter(a_true, a_pred, alpha=0.3, s=15, c='#f43f5e')
    ax5.plot([-1,1],[-1,1],'r--', lw=2)
    ax5.set_title(f'Arousal R2={r2_a:.3f}', color='#e2e8f0', fontweight='bold')
    ax5.set_xlim(-1.1,1.1); ax5.set_ylim(-1.1,1.1); ax5.set_aspect('equal')
    ax5.tick_params(colors='#e2e8f0'); ax5.grid(True, alpha=0.2, color='#334155')

    # Error histogram
    ax6 = fig.add_subplot(2, 3, 6, facecolor='#16213e')
    ax6.hist(v_pred-v_true, bins=40, alpha=0.6, color='#6366f1', label='Valence err', edgecolor='white', lw=0.5)
    ax6.hist(a_pred-a_true, bins=40, alpha=0.6, color='#f43f5e', label='Arousal err', edgecolor='white', lw=0.5)
    ax6.axvline(x=0, color='#fbbf24', ls='--', lw=2)
    ax6.set_title('Error Distribution', color='#e2e8f0', fontweight='bold')
    ax6.legend(fontsize=9); ax6.tick_params(colors='#e2e8f0')

    plt.tight_layout(rect=[0,0,1,0.95])
    report_path = os.path.join(os.path.dirname(__file__), 'accuracy_report.png')
    fig.savefig(report_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close(fig)
    print(f'\nReport saved: {report_path}')

    # 6. Save model in native format
    model_path = os.path.join(os.path.dirname(__file__), 'full_model.keras')
    model.save(model_path)
    print(f'Model saved: {model_path}')
    print('\nDone!')

if __name__ == '__main__':
    run()
