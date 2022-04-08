import glob
import pickle
import numpy as np
import tensorflow as tf
from datetime import date
from tensorflow import keras
from tensorflow.keras import layers, models
from multiprocessing import freeze_support
from src.misc_utils import create_folder, save_model_to_json
from src.CONSTS import (MAX_NUM_ATOMS, FEATURE_DEPTH)

today = str(date.today())


def core_model():
    '''
    mini unet
    '''
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH]
    inputs = layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.LayerNormalization(epsilon=1e-9)(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.LayerNormalization(epsilon=1e-9)(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.LayerNormalization(epsilon=1e-9)(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    for filters in [128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.LayerNormalization(epsilon=1e-9)(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.LayerNormalization(epsilon=1e-9)(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, 3]
    R_gen = layers.Conv2D(3, 3, activation=None, padding="same", use_bias=False)(x)
    # [BATCH, MAX_NUM_ATOMS, 3]
    R_gen = tf.reduce_mean(R_gen, axis=1)
    return inputs, R_gen


def loss_func(y_true_, y_pred):
    y_true_ = tf.cast(y_true_, tf.float32)
    mask = tf.cast(y_true_[..., 0] > 0, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    y_true = y_true_[..., 1:]
    y_pred *= mask
    # centre coordinates
    total_row = tf.reduce_sum(mask, axis=1, keepdims=True)
    centroid_t = tf.reduce_sum(y_true, axis=1, keepdims=True) / total_row
    centroid_g = tf.reduce_sum(y_pred, axis=1, keepdims=True) / total_row
    R_truth_c = (y_true - centroid_t) * mask
    R_gen_c = (y_pred - centroid_g) * mask

    # [BATCH, 3, 3]
    H = tf.linalg.matmul(R_gen_c, R_truth_c, transpose_a=True)
    _, U, V = tf.linalg.svd(H, full_matrices=True, compute_uv=True)

    # correct U
    det = tf.linalg.det(tf.matmul(V, U, transpose_b=True))
    diag_size = H.shape[1]
    batch_size = V.shape[0]
    det_eye = tf.tile(tf.eye(diag_size), tf.constant([batch_size, 1]))
    det_eye = tf.reshape(det_eye, [batch_size, diag_size, diag_size])
    det_eye = tf.cast(det_eye, tf.float32)
    indices = tf.where(det < 0)
    if indices.shape[0] > 0:
        det_re = tf.linalg.diag(tf.constant([1.0, 1.0, -1.0]))
        det_re = tf.tile(det_re, tf.constant([indices.shape[0], 1]))
        det_re = tf.reshape(det_re, [indices.shape[0], diag_size, diag_size])
        det_eye = tf.tensor_scatter_nd_update(det_eye, indices, det_re)
        V = tf.matmul(V, det_eye)

    # rotate and translate
    phi = tf.linalg.matmul(V, U, transpose_b=True)
    R_gen = tf.matmul(R_gen_c, phi, transpose_b=True) + centroid_t
    R_gen *= mask

    # return mse loss
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss = mse(y_true, R_gen)
    loss = tf.reduce_sum(loss, axis=-1)
    return loss


def get_optimizer(finetune=False):
    lr = 0.0001
    if finetune:
        lr = 0.00001
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [200000, 400000, 600000], [lr, lr / 10, lr / 50, lr / 100],
        name=None
    )
    opt_op = tf.keras.optimizers.Adam(learning_rate=lr_fn)
    return opt_op


def data_iterator(data_path):
    num_files = len(glob.glob(data_path + 'GR_*.pkl'))
    batch_nums = np.arange(num_files)
    while True:
        np.random.shuffle(batch_nums)
        for batch in batch_nums:
            f_name = data_path + 'GR_{}.pkl'.format(batch)
            with open(f_name, 'rb') as handle:
                GR = pickle.load(handle)

            G = GR[0].todense()
            d = G[..., -1]
            d = d + np.swapaxes(d, 1, 2)
            G[..., -1] = d
            R = GR[1].todense()
            if (G.shape[0] == 1) and (R.shape[0] != G.shape[0]):
                R = np.expand_dims(R, axis=0)

            sample_nums = np.arange(G.shape[0])
            np.random.shuffle(sample_nums)
            yield G[sample_nums, ...], R[sample_nums, ...]


def data_iterator_test(data_path):
    num_files = len(glob.glob(data_path + 'GR_*.pkl'))
    batch_nums = np.arange(num_files)
    for batch in batch_nums:
        f_name = data_path + 'GR_{}.pkl'.format(batch)
        with open(f_name, 'rb') as handle:
            GR = pickle.load(handle)

        G = GR[0].todense()
        d = G[..., -1]
        d = d + np.swapaxes(d, 1, 2)
        G[..., -1] = d
        R = GR[1].todense()
        if (G.shape[0] == 1) and (R.shape[0] != G.shape[0]):
            R = np.expand_dims(R, axis=0)

        yield G, R


if __name__ == "__main__":
    freeze_support()
    ckpt_path = 'checkpoints/generator_r_{}/'.format(today)
    create_folder(ckpt_path)
    create_folder("conf_model_r_{}".format(today))
    train_path = 'D:/seem_3d_data/train_data/train_batch/'
    val_path = 'D:/seem_3d_data/test_data/val_batch/'
    test_path = 'D:/seem_3d_data/test_data/test_batch/'

    train_steps = len(glob.glob(train_path + 'GR_*.pkl'))
    val_steps = len(glob.glob(val_path + 'GR_*.pkl'))

    for x, y in data_iterator(train_path):
        print(x.shape)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                    save_freq=1000,
                                                    save_weights_only=True,
                                                    monitor='loss',
                                                    mode='min',
                                                    save_best_only=True)]

    X, R_gen = core_model()

    model = model = keras.Model(inputs=X, outputs=R_gen)

    model.compile(optimizer=get_optimizer(),
                  loss=loss_func)

    save_model_to_json(model, "conf_model_r_{}/conf_model_r.json".format(today))
    model.summary()

    model.fit(data_iterator(train_path),
              epochs=4,
              validation_data=data_iterator(val_path),
              validation_steps=val_steps,
              callbacks=callbacks,
              steps_per_epoch=train_steps)
    res = model.evaluate(data_iterator_test(test_path),
                         return_dict=True)

    # save trained model in two ways
    model.save("conf_model_r_full_{}/".format(today))
    model_new = models.load_model("conf_model_r_full_{}/".format(today))
    res = model_new.evaluate(data_iterator_test(test_path),
                             return_dict=True)
