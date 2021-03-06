import os
import math
import json
import pickle
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
