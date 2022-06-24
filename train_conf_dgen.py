import glob
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from datetime import date
from tensorflow import keras
from tensorflow.keras import layers, models
from multiprocessing import freeze_support
from src.embed_utils import encoder_block
from src.misc_utils import create_folder, save_model_to_json, kabsch_fit, align_conf
from src.CONSTS import (MAX_NUM_ATOMS, FEATURE_DEPTH, NUM_COMPS, TF_EPS)

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=1000)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

today = str(date.today())
tfd = tfp.distributions


def core_model():
    '''
    mini resnet
    '''
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH]
    inputs = layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
    out = encoder_block(inputs, 64, pool=False)
    out = encoder_block(out, 64)
    out = encoder_block(out, 128, pool=False)
    out = encoder_block(out, 128)
    out = encoder_block(out, 256, pool=False)
    out = encoder_block(out, 256)
    out = encoder_block(out, 512, pool=False)
    out = encoder_block(out, 512)
    out = tf.keras.layers.GlobalMaxPooling2D()(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation("relu")(out)

    out = tf.keras.layers.Dense(MAX_NUM_ATOMS * 6, use_bias=False)(out)
    out = tf.reshape(out, [-1, MAX_NUM_ATOMS, 6])
    return inputs, out


def loss_func(y_true, y_pred):
    # [B,N,1]
    mask = tf.cast(tf.reduce_sum(y_true, axis=-1, keepdims=True) != 0, tf.float32)
    y_pred *= mask
    x_mean_std, y_mean_std, z_mean_std = tf.split(y_pred, 3, axis=-1)

    y_true_aligned = tf.py_function(align_conf,
                                    inp=[x_mean_std, y_mean_std, z_mean_std, y_true, mask],
                                    Tout=[tf.float32])

    x_mean = tf.expand_dims(x_mean_std[..., 0], axis=-1)
    x_log_std = tf.expand_dims(x_mean_std[..., 1], axis=-1)
    x_pdf = tfd.Normal(loc=x_mean, scale=tf.math.exp(x_log_std))
    aligned_x = tf.expand_dims(y_true_aligned[..., 0], axis=-1)
    x_log_density = tf.math.log(x_pdf.prob(aligned_x) + TF_EPS)

    y_mean = tf.expand_dims(y_mean_std[..., 0], axis=-1)
    y_log_std = tf.expand_dims(y_mean_std[..., 1], axis=-1)
    y_pdf = tfd.Normal(loc=y_mean, scale=tf.math.exp(y_log_std))
    aligned_y = tf.expand_dims(y_true_aligned[..., 1], axis=-1)
    y_log_density = tf.math.log(y_pdf.prob(aligned_y) + TF_EPS)

    z_mean = tf.expand_dims(z_mean_std[..., 0], axis=-1)
    z_log_std = tf.expand_dims(z_mean_std[..., 1], axis=-1)
    z_pdf = tfd.Normal(loc=z_mean, scale=tf.math.exp(z_log_std))
    aligned_z = tf.expand_dims(y_true_aligned[..., 2], axis=-1)
    z_log_density = tf.math.log(z_pdf.prob(aligned_z) + TF_EPS)

    # [BATCH, MAX_NUM_ATOMS, 1]
    _loss = x_log_density + y_log_density + z_log_density
    # [BATCH, MAX_NUM_ATOMS, 1]
    _loss *= mask
    loss = -tf.reduce_sum(_loss, axis=[1, 2])
    return loss


def distance_rmsd(y_true, y_pred):
    # [B,N,1]
    mask = tf.cast(tf.reduce_sum(y_true, axis=-1, keepdims=True) != 0, tf.float32)
    y_pred *= mask
    x_mean_std, y_mean_std, z_mean_std = tf.split(y_pred, 3, axis=-1)

    y_true_aligned = tf.py_function(align_conf,
                                    inp=[x_mean_std, y_mean_std, z_mean_std, y_true, mask],
                                    Tout=[tf.float32])
    y_pred_mean = tf.stack([x_mean_std[..., 0],
                            y_mean_std[..., 0],
                            z_mean_std[..., 0]], axis=-1)

    total_row = tf.reduce_sum(mask, axis=1, keepdims=True)
    loss = tf.math.squared_difference(y_pred_mean, y_true_aligned)
    loss = tf.reduce_sum(loss, axis=-1)
    loss = tf.reduce_sum(loss, axis=-1) / tf.squeeze(total_row)
    # [BATCH,]
    loss = tf.math.sqrt(loss)
    return loss


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def get_optimizer(finetune=False):
    lr = 0.0001
    if finetune:
        lr = 0.00001
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [1000000, 2000000, 3000000], [lr, lr / 10, lr / 50, lr / 100],
        name=None
    )
    opt_op = tf.keras.optimizers.Adam(learning_rate=lr_fn, global_clipnorm=0.5)
    return opt_op


def data_iterator(data_path):
    num_files = len(glob.glob(data_path + 'GDR_*.pkl'))
    batch_nums = np.arange(num_files)
    while True:
        np.random.shuffle(batch_nums)
        for batch in batch_nums:
            f_name = data_path + 'GDR_{}.pkl'.format(batch)
            with open(f_name, 'rb') as handle:
                GD = pickle.load(handle)

            G = GD[0].todense()
            R = GD[1].todense()

            sample_nums = np.arange(G.shape[0])
            np.random.shuffle(sample_nums)
            yield G[sample_nums, ...], R[sample_nums, ...]


def data_iterator_test(data_path):
    num_files = len(glob.glob(data_path + 'GDR_*.pkl'))
    batch_nums = np.arange(num_files)
    for batch in batch_nums:
        f_name = data_path + 'GDR_{}.pkl'.format(batch)
        with open(f_name, 'rb') as handle:
            GD = pickle.load(handle)

        G = GD[0].todense()
        R = GD[1].todense()
        yield G, R


if __name__ == "__main__":
    freeze_support()
    ckpt_path = 'checkpoints/generator_d_K_{}/'.format(NUM_COMPS)
    create_folder(ckpt_path)
    create_folder("conf_model_d_K_{}".format(NUM_COMPS))
    train_path = 'D:/seem_3d_data/train_data/train_batch/'
    val_path = 'D:/seem_3d_data/test_data/val_batch/'
    test_path = 'D:/seem_3d_data/test_data/test_batch/'

    f_name = train_path + 'stats.pkl'
    with open(f_name, 'rb') as handle:
        d_mean, d_std = pickle.load(handle)

    train_steps = len(glob.glob(train_path + 'GDR_*.pkl'))
    val_steps = len(glob.glob(val_path + 'GDR_*.pkl'))

    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                    save_freq=1000,
                                                    save_weights_only=True,
                                                    monitor='loss',
                                                    mode='min',
                                                    save_best_only=True)]

    X, logits = core_model()

    model = keras.Model(inputs=X, outputs=logits)
    breakpoint()

    model.compile(optimizer=get_optimizer(),
                  loss=loss_func, metrics=[distance_rmse])

    save_model_to_json(model, "conf_model_d_K_{}/conf_model_d.json".format(NUM_COMPS))
    model.summary()
    breakpoint()

    try:
        model.load_weights("./checkpoints/generator_d_K_1/")
    except:
        print('no exitsing model detected, training starts afresh')
        pass

    model.fit(data_iterator(train_path),
              epochs=40,
              validation_data=data_iterator(val_path),
              validation_steps=val_steps,
              callbacks=callbacks,
              steps_per_epoch=train_steps)
    res = model.evaluate(data_iterator_test(test_path),
                         return_dict=True)

    # save trained model in two ways
    model.save("conf_model_d_full_K_{}/".format(NUM_COMPS))
    model_new = models.load_model("conf_model_d_full_K_{}/".format(NUM_COMPS))
    res = model_new.evaluate(data_iterator_test(test_path),
                             return_dict=True)
