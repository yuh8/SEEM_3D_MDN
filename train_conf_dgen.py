import glob
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import callbacks
from multiprocessing import freeze_support
from src.embed_utils import get_g_net
from src.misc_utils import create_folder
from src.CONSTS import (MAX_NUM_ATOMS, FEATURE_DEPTH, BATCH_SIZE, VAL_BATCH_SIZE,MAX_EPOCH)

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=1000)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def get_metrics():
    aal = keras.metrics.Mean(name="avg_abs_loss")
    return aal


def core_model():
    inputs = keras.layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 4))
    # (batch_size, num_atoms, d_model)
    h = g_net(inputs)
    h = tf.reduce_mean(h, axis=1)
    # (batch_size, 3)
    y_prop = tf.keras.layers.Dense(3)(h)
    return inputs, y_prop


def loss_func_r(y_true, y_pred):
    # [B,3]
    abs_error = tf.abs(y_true - y_pred)
    avg_abs_error = tf.reduce_mean(abs_error, axis=-1)
    return avg_abs_error


class WarmDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps=4000):
        super(WarmDecay, self).__init__()
        self.d_model = 9612
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def get_optimizer():
    opt_op = tf.keras.optimizers.Adam(learning_rate=WarmDecay(), clipnorm=1)
    return opt_op


class TransVAE(Model):
    def compile(self, optimizer, metrics):
        super(TransVAE, self).compile()
        self.optimizer = optimizer
        self.aal = metrics[0]

    def train_step(self, data):
        X = data[0]
        y_true = data[1]

        # capture the scope of gradient
        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)
            loss = loss_func_r(y_true, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.aal.update_state(loss)
        return {"avs_error": self.aal.result()}

    def test_step(self, data):
        X = data[0]
        y_true = data[1]
        y_pred = self(X, training=False)
        loss = loss_func_r(y_true, y_pred)
        self.aal.update_state(loss)
        return {"avs_error": self.aal.result()}

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.kl, self.r_rmsd]


def data_iterator_train():
    num_files = len(glob.glob(train_path + 'GDR_*.npz'))
    batch_nums = np.arange(num_files)
    np.random.shuffle(batch_nums)
    while True:
        np.random.shuffle(batch_nums)
        for batch in batch_nums:
            f_name = train_path + f'GDR_{batch}.npz'
            GDR = np.load(f_name)
            G = GDR['G']
            R = GDR['R']
            yield G, R


def data_iterator_val():
    num_files = len(glob.glob(val_path + 'GDR_*.npz'))
    batch_nums = np.arange(num_files)
    while True:
        np.random.shuffle(batch_nums)
        for batch in batch_nums:
            f_name = val_path + f'GDR_{batch}.npz'
            GDR = np.load(f_name)
            G = GDR['G']
            R = GDR['R']
            yield G, R


def data_iterator_test():
    num_files = len(glob.glob(test_path + 'GDR_*.npz'))
    batch_nums = np.arange(num_files)
    for batch in batch_nums:
        f_name = test_path + f'GDR_{batch}.npz'
        GDR = np.load(f_name)
        G = GDR['G']
        R = GDR['R']
        yield G, R


def _fixup_shape(x, y):
    x.set_shape([None, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 4])
    y.set_shape([None, 3])
    return x, y


if __name__ == "__main__":
    freeze_support()
    ckpt_path = 'checkpoints/TransVAE_qm9_prop/'
    create_folder(ckpt_path)
    create_folder("g_net_qm9_prop")
    train_path = '/mnt/transvae_qm9_prop/train_data/train_batch/'
    val_path = '/mnt/transvae_qm9_prop/test_data/val_batch/'
    test_path = '/mnt/transvae_qm9_prop/test_data/test_batch/'

    train_steps = len(glob.glob(train_path + 'GDR_*.npz')) // BATCH_SIZE
    val_steps = len(glob.glob(val_path + 'GDR_*.npz')) // VAL_BATCH_SIZE

    # get models
    g_net = get_g_net()

    # callbacks
    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                    save_freq=1000,
                                                    save_weights_only=True),
                 tf.keras.callbacks.TensorBoard('./logs_transvae_qm9_prop', update_freq=10)]

    # compile model
    X, y_pred = core_model()
    transvae = TransVAE(inputs=X, outputs=y_pred)
    optimizer = get_optimizer()
    transvae.compile(optimizer=get_optimizer(), metrics=[get_metrics()])
    transvae.summary()

    try:
        transvae.load_weights("./checkpoints/TransVAE_qm9_prop/")
    except:
        print('no exitsing model detected, training starts afresh')
        pass

    train_dataset = tf.data.Dataset.from_generator(
        data_iterator_train,
        output_types=(tf.float32, tf.float32),
        output_shapes=((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 4), 3))

    train_dataset = train_dataset.shuffle(buffer_size=1000, seed=0,
                                          reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True).map(_fixup_shape)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        data_iterator_val,
        output_types=(tf.float32, tf.float32),
        output_shapes=((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 4), 3))
    val_dataset = val_dataset.batch(VAL_BATCH_SIZE, drop_remainder=True).map(_fixup_shape)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_generator(
        data_iterator_test,
        output_types=(tf.float32, tf.float32),
        output_shapes=((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 4), 3))
    test_dataset = test_dataset.batch(VAL_BATCH_SIZE, drop_remainder=True).map(_fixup_shape)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    transvae.fit(train_dataset,
                 epochs=MAX_EPOCH,
                 validation_data=val_dataset,
                 validation_steps=val_steps,
                 callbacks=callbacks,
                 steps_per_epoch=train_steps)
    res = transvae.evaluate(test_dataset,
                            return_dict=True)

    # save trained model
    g_net.compile(optimizer='SGD', loss=None)
    g_net.save('g_net_qm9_prop/' + 'GNet')
