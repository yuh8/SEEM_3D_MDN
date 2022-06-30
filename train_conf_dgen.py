import glob
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from datetime import date
from tensorflow import keras
from tensorflow.keras import layers, models
from multiprocessing import freeze_support
from src.embed_utils import encoder_block, decoder_block, conv2d_block
from src.misc_utils import create_folder, save_model_to_json
from src.CONSTS import (MAX_NUM_ATOMS, FEATURE_DEPTH, NUM_COMPS, OUTPUT_DEPTH, TF_EPS)

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
    mini unet
    '''
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH]
    inputs = layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
    s1, p1 = encoder_block(inputs, 64, pool=False)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128, pool=False)
    s4, p4 = encoder_block(p3, 128)
    s5, p5 = encoder_block(p4, 256, pool=False)
    s6, p6 = encoder_block(p5, 256)
    s7, p7 = encoder_block(p6, 384)

    b1 = conv2d_block(p7, 512)

    d1 = decoder_block(b1, s7, 384)
    d2 = decoder_block(d1, s6, 256)
    d3 = decoder_block(d2, s5, 256, unpool=False)
    d4 = decoder_block(d3, s4, 128)
    d5 = decoder_block(d4, s3, 128, unpool=False)
    d6 = decoder_block(d5, s2, 64)
    d7 = decoder_block(d6, s1, 64, unpool=False)

    # Add a per-pixel classification layer
    logits = layers.Conv2D(OUTPUT_DEPTH, 1, activation=None, padding="same", use_bias=False)(d7)
    return inputs, logits


def loss_func(y_true, y_pred):
    # compute only for upper triangle
    # y_true = tf.linalg.band_part(y_true, 0, -1)
    comp_weight, mean, log_std = tf.split(y_pred, 3, axis=-1)
    comp_weight = tf.nn.softmax(comp_weight, axis=-1)
    dist = tfd.Normal(loc=mean, scale=tf.math.exp(log_std))
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, NUM_COMPS]
    _loss = comp_weight * dist.prob(y_true)
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS]
    _loss = tf.reduce_sum(_loss, axis=-1)
    _loss = tf.math.log(_loss + TF_EPS)
    mask = tf.squeeze(tf.cast(y_true != 0, tf.float32))
    _loss *= mask
    loss = -tf.reduce_sum(_loss, axis=[1, 2])
    return loss


def distance_rmse(y_true, y_pred):
    # compute only for upper triangle
    # y_true = tf.linalg.band_part(y_true, 0, -1)
    comp_weight, mean, _ = tf.split(y_pred, 3, axis=-1)
    comp_weight = tf.nn.softmax(comp_weight, axis=-1)
    se = (mean - y_true)**2
    se *= comp_weight
    se = tf.reduce_sum(se, axis=-1)
    mask = tf.squeeze(tf.cast(y_true != 0, tf.float32))
    se *= mask
    loss = tf.reduce_sum(se, axis=[1, 2]) / tf.reduce_sum(mask, axis=[1, 2])
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
        [40000, 80000, 120000], [lr, lr / 2, lr / 4, lr / 10],
        name=None
    )
    opt_op = tf.keras.optimizers.Adam(learning_rate=lr_fn, clipnorm=0.5)
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
            D = GD[1].todense()
            # D -= d_mean
            # D /= d_std
            mask = G.sum(-1) > 3
            D *= mask

            sample_nums = np.arange(G.shape[0])
            np.random.shuffle(sample_nums)
            yield G[sample_nums, ...], np.expand_dims(D[sample_nums, ...], axis=-1)


def data_iterator_test(data_path):
    num_files = len(glob.glob(data_path + 'GDR_*.pkl'))
    batch_nums = np.arange(num_files)
    for batch in batch_nums:
        f_name = data_path + 'GDR_{}.pkl'.format(batch)
        with open(f_name, 'rb') as handle:
            GD = pickle.load(handle)

        G = GD[0].todense()
        D = GD[1].todense()
        # D -= d_mean
        # D /= d_std
        mask = G.sum(-1) > 3
        D *= mask
        yield G, np.expand_dims(D, axis=-1)


if __name__ == "__main__":
    freeze_support()
    ckpt_path = 'checkpoints/generator_d_K_{}/'.format(NUM_COMPS)
    create_folder(ckpt_path)
    create_folder("conf_model_d_K_{}".format(NUM_COMPS))
    train_path = '/mnt/seem_3d_data/train_data/train_batch/'
    val_path = '/mnt/seem_3d_data/test_data/val_batch/'
    test_path = '/mnt/seem_3d_data/test_data/test_batch/'

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

    model = model = keras.Model(inputs=X, outputs=logits)

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
              epochs=100,
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
