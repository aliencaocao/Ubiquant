import gc
import logging

logging.disable(logging.WARNING)
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import plot_model
import tensorflow_probability as tfp

mixed_precision.set_global_policy('mixed_float16')
print(tf.__version__)

# Load data
# df = pd.read_parquet('E:/train_low_mem.parquet')
df = pd.read_csv('kaggle/input/ubiquant-market-prediction/train.csv', engine='pyarrow')
df.drop(['row_id'], inplace=True, axis=1)
investment_id = df['investment_id']
df = df.groupby(['time_id'])
x, x_id, y = [], [], []
for timestep in df:
    timestep = timestep[1]  # remove pd added index tuple
    x.append(timestep.loc[:, 'f_0':'f_299'].to_numpy())
    x_id.append(timestep['investment_id'].to_numpy())
    y.append(timestep['target'].to_numpy())

investment_ids = list(investment_id.unique())
investment_id_lookup_layer = IntegerLookup(oov_token=-1, output_mode='int')
investment_id_lookup_layer.adapt(np.array(investment_ids))
vocab = investment_id_lookup_layer.get_vocabulary(include_special_tokens=True)
vocab_size = investment_id_lookup_layer.vocabulary_size()
print(max(investment_ids))
print(vocab_size)
assert len(x) == len(y)


def preprocess(x, x_id, y):
    global x_time, width, num_remaining, remaining_slice, rows_to_fill, padded_remaining
    x_time = [tf.convert_to_tensor(s, dtype=tf.float16) for s in x]
    width = x_time[0].shape[1]  # should be 300 (no of features)
    num_remaining = [s.shape[0] % width for s in x_time]  # remaining rows less than 300
    remaining_slice = [x_time[i][-num_remaining[i]:] for i in range(len(x_time))]
    rows_to_fill = [width - r for r in num_remaining]
    padded_remaining = [tf.concat([remaining_slice[i], tf.cast(tf.zeros([rows_to_fill[i], width]), tf.float16)], axis=0) for i in range(len(x_time))]
    x_time = [tf.concat([x_time[i], padded_remaining[i]], axis=0) for i in range(len(x_time))]  # concat padded remaining rows to already split rows
    x_time = [tf.reshape(s[:-(s.shape[0] % width)], (width, width, -1)) for s in x_time]  # reshape automatically split rows
    x_time = [[x_t] * len(s) for x_t, s in zip(x_time, x_id)]
    return (x, x_id, x_time), y  # x_time is a list of same images of n x 300 x 300, same across a time step


# Pearson correlation coefficient loss and metrics
def pearson_corr(y_true, y_pred, axis=-1):
    return tfp.stats.correlation(tf.squeeze(y_pred), tf.squeeze(y_true), sample_axis=axis, event_axis=None)


def pearson_corr_loss(y_true, y_pred, axis=-1):
    return (1 - pearson_corr(y_true, y_pred, axis=axis)) + 2 * tf.keras.metrics.mean_squared_error(y_true, y_pred)


print('Preprocessing dataset')
ds = preprocess(x, x_id, y)
print('Done')

print(ds[0][2][-1][-1].shape)
np.save('ds_20000', ds)
