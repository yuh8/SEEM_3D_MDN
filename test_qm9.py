import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import glob
import numpy as np
import tensorflow as tf
from multiprocessing import freeze_support
from src.misc_utils import pickle_load


def _standardize_prop(r, mean_std):
    r[0] = (r[0] - mean_std[0]) / mean_std[1]
    r[1] = (r[1] - mean_std[2]) / mean_std[3]
    r[2] = (r[2] - mean_std[4]) / mean_std[5]
    return r


def _destandardize_prop(r, mean_std):
    r[0] = r[0] * mean_std[1] + mean_std[0]
    r[1] = r[1] * mean_std[3] + mean_std[2]
    r[2] = r[2] * mean_std[5] + mean_std[4]
    return r


def load_models():
    g_net = tf.keras.models.load_model('g_net_qm9_prop_scaffold/GNet/')
    return g_net


def get_prediction(g):
    g = np.expand_dims(g, axis=0)

    with tf.device('/cpu:2'):
        y_pred = g_net.predict(g)[0]

    y_pred = _destandardize_prop(y_pred, mean_std)
    return y_pred


def data_iterator_test():
    num_files = len(glob.glob(test_path + 'GDR_*.npz'))
    batch_nums = np.arange(num_files)
    for batch in batch_nums:
        f_name = test_path + f'GDR_{batch}.npz'
        GDR = np.load(f_name)
        G = GDR['G']
        Y = GDR['Y']
        Y = _standardize_prop(Y, mean_std)
        yield G, Y


def compute_mae():
    diffs = []
    idx = 0
    for g, y in data_iterator_test():
        y_pred = get_prediction(g)
        y_true = _destandardize_prop(y, mean_std)
        avg_diff = np.abs(y_true - y_pred).sum() / 3
        idx += 1
        if idx % 100 == 0:
            print(f'mean average error is {np.mean(diffs)}')
        diffs.append(avg_diff)

    print(f'mean average error is {np.mean(diffs)}')
    print(f'std of error is {np.std(diffs)}')


if __name__ == "__main__":
    freeze_support()
    g_net = load_models()
    train_path = '/mnt/transvae_qm9_prop_scaffold/train_data/train_batch/'
    test_path = '/mnt/transvae_qm9_prop_scaffold/test_data/test_batch/'
    mean_std = pickle_load(train_path + 'stats.pkl')
    compute_mae()
