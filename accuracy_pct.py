"""Compute accuracy in percentage terms, write to file."""
import numpy as np
import tensorflow as tf
from model_training import generate_synthetic_data, concordance_cc

model = tf.keras.models.load_model('full_model.keras')

au_data, audio_data, labels = generate_synthetic_data(n_samples=5000)
np.random.seed(42)
idx = np.random.permutation(len(labels))
test_idx = idx[int(0.8*len(labels)):]

au_test = au_data[test_idx]
audio_test = audio_data[test_idx]
y_test = labels[test_idx]

y_pred = model.predict([au_test, audio_test], verbose=0)
v_true, a_true = y_test[:, 0], y_test[:, 1]
v_pred, a_pred = y_pred[:, 0], y_pred[:, 1]

lines = []
lines.append('=' * 60)
lines.append('  MODEL ACCURACY IN PERCENTAGE')
lines.append('=' * 60)
lines.append('')

for thresh in [0.1, 0.2, 0.3, 0.5]:
    v_acc = np.mean(np.abs(v_pred - v_true) <= thresh) * 100
    a_acc = np.mean(np.abs(a_pred - a_true) <= thresh) * 100
    avg = (v_acc + a_acc) / 2
    lines.append(f'  Within +/-{thresh:.1f} :  Valence {v_acc:.1f}%  |  Arousal {a_acc:.1f}%  |  Avg {avg:.1f}%')

mae_v = np.mean(np.abs(v_pred - v_true))
mae_a = np.mean(np.abs(a_pred - a_true))
norm_acc_v = (1 - mae_v / 2) * 100
norm_acc_a = (1 - mae_a / 2) * 100
norm_acc_avg = (norm_acc_v + norm_acc_a) / 2

lines.append('')
lines.append('  Normalized Accuracy:')
lines.append(f'    Valence : {norm_acc_v:.1f}%')
lines.append(f'    Arousal : {norm_acc_a:.1f}%')
lines.append(f'    Average : {norm_acc_avg:.1f}%')

def classify_zone(v, a):
    if a > 0.3 and v < -0.2: return 'high_stress'
    elif a > 0.5 and v < 0.0: return 'anxiety'
    elif a > 0.4 and v < -0.3: return 'anger'
    elif a < -0.1 and v < -0.2: return 'sadness'
    elif a > 0.3 and v > 0.3: return 'joy'
    elif a < 0.1 and v > 0.1: return 'calm'
    else: return 'neutral'

true_zones = [classify_zone(v, a) for v, a in zip(v_true, a_true)]
pred_zones = [classify_zone(v, a) for v, a in zip(v_pred, a_pred)]
zone_acc = np.mean(np.array(true_zones) == np.array(pred_zones)) * 100

lines.append('')
lines.append(f'  Zone Classification Accuracy: {zone_acc:.1f}%')
lines.append('  Zone-wise Breakdown:')

from collections import Counter
zone_counts = Counter(true_zones)
for zone in sorted(zone_counts.keys()):
    mask = np.array(true_zones) == zone
    correct = np.sum(np.array(pred_zones)[mask] == zone)
    total = int(mask.sum())
    acc = (correct / total * 100) if total > 0 else 0
    lines.append(f'    {zone:<15}: {correct}/{total} correct = {acc:.1f}%')

r2_v = 1 - np.sum((v_true - v_pred)**2) / np.sum((v_true - np.mean(v_true))**2)
r2_a = 1 - np.sum((a_true - a_pred)**2) / np.sum((a_true - np.mean(a_true))**2)

lines.append('')
lines.append('  Variance Explained:')
lines.append(f'    Valence : {r2_v*100:.1f}%')
lines.append(f'    Arousal : {r2_a*100:.1f}%')
lines.append(f'    Average : {(r2_v+r2_a)/2*100:.1f}%')

lines.append('')
lines.append('=' * 60)
lines.append('  SUMMARY')
lines.append('=' * 60)
pred_02 = (np.mean(np.abs(v_pred-v_true)<=0.2)*100 + np.mean(np.abs(a_pred-a_true)<=0.2)*100)/2
lines.append(f'  Prediction within +/-0.2  : {pred_02:.1f}%')
lines.append(f'  Normalized Accuracy       : {norm_acc_avg:.1f}%')
lines.append(f'  Zone Classification       : {zone_acc:.1f}%')
lines.append(f'  Variance Explained        : {(r2_v+r2_a)/2*100:.1f}%')
lines.append('=' * 60)

result = '\n'.join(lines)
with open('accuracy_results.txt', 'w', encoding='utf-8') as f:
    f.write(result)
print('Results written to accuracy_results.txt')
