import glob
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from datetime import date
from tensorflow import keras
from tensorflow.keras import layers, models
from multiprocessing import freeze_support
from src.misc_utils import create_folder, save_model_to_json, norm_pdf
from src.CONSTS import (MAX_NUM_ATOMS, FEATURE_DEPTH, NUM_COMPS, OUTPUT_DEPTH, TF_EPS)

today = str(date.today())
tfd = tfp.distributions


def core_model():
    '''
    mini unet
    '''
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH]
    inputs = layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH - 1))
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.LayerNormalization(epsilon=1e-9)(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
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

    for filters in [256, 128, 64, 32]:
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

    # Add a per-pixel classification layer
    logits = layers.Conv2D(OUTPUT_DEPTH, 3, activation=None, padding="same", use_bias=False)(x)
    return inputs, logits


def loss_func(y_true, y_pred):
    comp_weight, mean, log_std = tf.split(y_pred, 3, axis=-1)
    comp_weight = tf.nn.softmax(comp_weight, axis=-1)
    dist = tfd.Normal(loc=mean, scale=tf.math.exp(log_std))
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, NUM_COMPS]
    _loss = comp_weight * dist.prob(y_true)
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS]
    _loss = tf.reduce_sum(_loss, axis=-1)
    _loss = tf.math.log(_loss + TF_EPS)
    mask = tf.squeeze(tf.cast(y_true > 0, tf.float32))
    _loss *= mask
    loss = -tf.reduce_sum(_loss, axis=[1, 2])
    return loss


def distance_loss(y_true, y_pred):
    comp_weight, mean, _ = tf.split(y_pred, 3, axis=-1)
    comp_weight = tf.nn.softmax(comp_weight, axis=-1)
    se = (mean - y_true)**2
    se *= comp_weight
    se = tf.reduce_sum(se, axis=-1)
    mask = tf.squeeze(tf.cast(y_true > 0, tf.float32))
    se *= mask
    loss = tf.reduce_sum(se, axis=[1, 2])
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


def get_optimizer():
    lr_fn = CustomSchedule(512)
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
            # R = GR[1].todense()
            sample_nums = np.arange(G.shape[0])
            np.random.shuffle(sample_nums)
            yield G[sample_nums, ..., :-1], np.expand_dims(G[sample_nums, ..., -1], axis=-1)


def data_iterator_test(data_path):
    num_files = len(glob.glob(data_path + 'GR_*.pkl'))
    batch_nums = np.arange(num_files)
    for batch in batch_nums:
        f_name = data_path + 'GR_{}.pkl'.format(batch)
        with open(f_name, 'rb') as handle:
            GR = pickle.load(handle)

        G = GR[0].todense()
        # R = GR[1].todense()
        yield G[..., :-1], np.expand_dims(G[..., -1], axis=-1)


if __name__ == "__main__":
    freeze_support()
    ckpt_path = 'checkpoints/generator_d_K_{}/'.format(NUM_COMPS)
    create_folder(ckpt_path)
    create_folder("conf_model_d_K_{}".format(NUM_COMPS))
    train_path = 'D:/seem_3d_data/train_data/train_batch/'
    val_path = 'D:/seem_3d_data/test_data/val_batch/'
    test_path = 'D:/seem_3d_data/test_data/test_batch/'

    train_steps = len(glob.glob(train_path + 'GR_*.pkl'))
    val_steps = len(glob.glob(val_path + 'GR_*.pkl'))

    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                    save_freq=1000,
                                                    save_weights_only=True,
                                                    monitor='loss',
                                                    mode='min',
                                                    save_best_only=True)]

    X, logits = core_model()

    model = model = keras.Model(inputs=X, outputs=logits)

    model.compile(optimizer=get_optimizer(),
                  loss=loss_func, metrics=[distance_loss])

    save_model_to_json(model, "conf_model_d_K_{}/conf_model_d.json".format(NUM_COMPS))
    model.summary()

    model.fit(data_iterator(train_path),
              epochs=20,
              validation_data=data_iterator(val_path),
              validation_steps=val_steps,
              callbacks=callbacks,
              steps_per_epoch=train_steps)
    res = model.evaluate(data_iterator_test(test_path),
                         return_dict=True)

    # save trained model in two ways
    model.save("conf_model_d_full_{}/".format(today))
    model_new = models.load_model("conf_model_d_full_K_{}/".format(NUM_COMPS))
    res = model_new.evaluate(data_iterator_test(test_path),
                             return_dict=True)
