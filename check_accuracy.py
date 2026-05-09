"""Quick accuracy check on trained model."""
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model_training import build_full_model, generate_synthetic_data, concordance_cc

model = tf.keras.models.load_model(
    'full_model.h5',
    custom_objects={
        'combined_ccc_mse_loss': lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred)),
        'concordance_cc': concordance_cc,
    }
)
print('Model loaded successfully!')
print('Parameters:', f'{model.count_params():,}')

au_data, audio_data, labels = generate_synthetic_data(n_samples=5000)
n = len(labels)
np.random.seed(42)
idx = np.random.permutation(n)
test_idx = idx[int(0.8*n):]

au_test = au_data[test_idx]
audio_test = audio_data[test_idx]
y_test = labels[test_idx]

y_pred = model.predict([au_test, audio_test], verbose=0)
v_true, a_true = y_test[:, 0], y_test[:, 1]
v_pred, a_pred = y_pred[:, 0], y_pred[:, 1]

ccc_v = float(concordance_cc(tf.constant(v_true, dtype=tf.float32), tf.constant(v_pred, dtype=tf.float32)).numpy())
ccc_a = float(concordance_cc(tf.constant(a_true, dtype=tf.float32), tf.constant(a_pred, dtype=tf.float32)).numpy())
mae_v = mean_absolute_error(v_true, v_pred)
mae_a = mean_absolute_error(a_true, a_pred)
mse_v = mean_squared_error(v_true, v_pred)
mse_a = mean_squared_error(a_true, a_pred)
r2_v = r2_score(v_true, v_pred)
r2_a = r2_score(a_true, a_pred)
pearson_v = float(np.corrcoef(v_true, v_pred)[0, 1])
pearson_a = float(np.corrcoef(a_true, a_pred)[0, 1])

print()
print('=' * 60)
print('  MODEL ACCURACY REPORT - Test Set:', len(y_test), 'samples')
print('=' * 60)
header = '{:<30} {:>12} {:>12}'.format('Metric', 'Valence', 'Arousal')
print(header)
print('-' * 55)
print('{:<30} {:>12.4f} {:>12.4f}'.format('CCC - Concordance', ccc_v, ccc_a))
print('{:<30} {:>12.4f} {:>12.4f}'.format('Pearson Correlation', pearson_v, pearson_a))
print('{:<30} {:>12.4f} {:>12.4f}'.format('R2 Score', r2_v, r2_a))
print('{:<30} {:>12.4f} {:>12.4f}'.format('MAE - Mean Abs Error', mae_v, mae_a))
print('{:<30} {:>12.4f} {:>12.4f}'.format('MSE - Mean Sq Error', mse_v, mse_a))
print('{:<30} {:>12.4f} {:>12.4f}'.format('RMSE - Root MSE', np.sqrt(mse_v), np.sqrt(mse_a)))
print('-' * 55)

avg_ccc = (ccc_v + ccc_a) / 2
if avg_ccc >= 0.70: quality = 'EXCELLENT'
elif avg_ccc >= 0.50: quality = 'GOOD'
elif avg_ccc >= 0.30: quality = 'MODERATE'
elif avg_ccc >= 0.10: quality = 'POOR'
else: quality = 'VERY POOR'

print('{:<30} {:>12.4f}'.format('Average CCC', avg_ccc))
print('{:<30} {:>12}'.format('Overall Quality', quality))
print('=' * 60)
