import os
import math
import json
import pickle
import numpy as np
import tensorflow as tf
from .CONSTS import TF_EPS


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def pickle_save(file, save_path):
    with open(save_path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(load_path):
    with open(load_path, 'rb') as handle:
        file = pickle.load(handle)
    return file


def save_model_to_json(model, model_path):
    model_json = model.to_json()
    with open("{}".format(model_path), "w") as json_file:
        json.dump(model_json, json_file)


def load_json_model(model_path, custom_obj=None, custom_obj_name=None):
    with open("{}".format(model_path)) as json_file:
        model_json = json.load(json_file)
    if custom_obj is not None:
        uncompiled_model = tf.keras.models.model_from_json(model_json,
                                                           {custom_obj_name: custom_obj})
    else:
        uncompiled_model = tf.keras.models.model_from_json(model_json)
    return uncompiled_model


def norm_pdf(x, mu, log_var):
    '''
    x : [BATCH, MAX_ATOM_NUM, MAX_ATOM_NUM, 1]
    '''
    var = tf.math.exp(log_var)
    z = (2 * math.pi * var)**0.5 + TF_EPS
    pdf = tf.math.exp(-0.5 * (x - mu)**2 / var) / z
    return pdf


class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())


def centroid(X, mask):
    # [B,1,1]
    total_row = np.sum(mask, axis=1, keepdims=True)
    # [B,1,D]
    C = np.sum(X, axis=1, keepdims=True) / total_row
    return C


def kabsch(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (B,N,D) matrix, where B is batch size, N is points and D is dimension.
    Q : array
        (B,N,D) matrix, where B is batch size, N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (B,D,D)
    """

    # [B,D,D]
    C = np.matmul(np.transpose(P, [2, 1]), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    U, _, VT = np.linalg.svd(C)
    d = (np.linalg.det(VT) * np.linalg.det(U)) < 0.0
    V = np.transpose(VT, [2, 1])
    V[d, ..., -1] = -V[d, ..., -1]

    # Create Rotation matrix U
    Uh = np.transpose(U, [2, 1])
    R = np.matmul(V, Uh)
    return R


def kabsch_rotate(P, Q):
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm.
    Parameters
    ----------
    P : array
        (B,N,D) matrix, where B is batch size, N is points and D is dimension.
    Q : array
        (B,N,D) matrix, where B is batch size, N is points and D is dimension.
    Returns
    -------
    P : array
        (B,N,D) matrix, where B is batch size, N is points and D is dimension,
        rotated
    """
    R = kabsch(P, Q)
    Rh = np.transpose(R, [2, 1])

    # Rotate P [B,N,D] * [B,D,D]
    P = np.matmul(P, Rh)
    return P


def kabsch_fit(P, Q, mask):
    '''
    P: [B,N,D]
    Q: [B,N,D]

    mask: [B,N,1]
    '''
    QC = centroid(Q, mask)
    Q = (Q - QC) * mask
    P = (P - centroid(P, mask)) * mask
    P = (kabsch_rotate(P, Q) + QC) * mask
    return P


def align_conf(x_mean_std, y_mean_std, z_mean_std, y_true, mask):
    x_mean = np.expand_dims(x_mean_std.numpy()[..., 0], axis=-1)
    y_mean = np.expand_dims(y_mean_std.numpy()[..., 0], axis=-1)
    z_mean = np.expand_dims(z_mean_std.numpy()[..., 0], axis=-1)
    _y_pred = np.concatenate((x_mean, y_mean, z_mean), axis=-1)
    # rotate true on predicted coordinates
    y_true_aligned = kabsch_fit(y_true.numpy(), _y_pred, mask.numpy()).astype(np.float32)
    return tf.convert_to_tensor(y_true_aligned)
