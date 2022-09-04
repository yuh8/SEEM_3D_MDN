import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from multiprocessing import freeze_support
from train_conf_dgen import data_iterator_test
from src.misc_utils import pickle_load


def _destandardize_prop(r, mean_std):
    r[0] = r[0] * mean_std[1] + mean_std[0]
    r[1] = r[1] * mean_std[3] + mean_std[2]
    r[2] = r[2] * mean_std[5] + mean_std[4]
    return r


def load_models():
    g_net = tf.keras.models.load_model('g_net_qm9_prop/GNet/')
    return g_net


def get_prediction(g):
    g = np.expand_dims(g, axis=0)

    with tf.device('/cpu:2'):
        y_pred = g_net.predict(g)[0]

    y_pred = _destandardize_prop(y_pred, mean_std)
    return y_pred


def compute_mae():
    diffs = []
    for g, y in data_iterator_test():
        y_pred = get_prediction(g)
        y_true = _destandardize_prop(y, mean_std)
        avg_diff = np.abs(y_true - y_pred).sum() / 3
        diffs.append(avg_diff)

    print(f'mean average error is {np.mean(diffs)}')


if __name__ == "__main__":
    freeze_support()
    g_net = load_models()
    train_path = '/mnt/transvae_qm9_prop/train_data/train_batch/'
    test_path = '/mnt/transvae_qm9_prop/test_data/test_batch/'
    mean_std = pickle_load(train_path + 'stats.pkl')
    compute_mae()
