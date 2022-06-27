import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from .CONSTS import (MAX_NUM_ATOMS, FEATURE_DEPTH, HIDDEN_SIZE)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def reparameterize(mean, logvar):
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * logvar) * epsilon


def bottleneck(input, num_filters, kernel_size=3, padding='SAME'):
    x = tf.keras.layers.Conv2D(num_filters // 4,
                               kernel_size=1,
                               padding=padding)(input)
    x = tf.keras.layers.Conv2D(num_filters // 4,
                               kernel_size=kernel_size,
                               padding=padding)(x)
    x = tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding=padding)(x)
    return x


def conv2d_block(input, num_filters, kernel_size=3, padding='SAME'):
    if num_filters >= 256:
        x = bottleneck(input, num_filters)
    else:
        x = tf.keras.layers.Conv2D(num_filters,
                                   kernel_size=kernel_size,
                                   padding=padding)(input)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x


def encoder_block(X, num_filters, pool=True):
    X = conv2d_block(X, num_filters)
    X = conv2d_block(X, num_filters // 2)
    X = conv2d_block(X, num_filters)
    if pool:
        p = tf.keras.layers.MaxPool2D(2, 2)(X)
    else:
        p = X
    return X, p


def decoder_block(X, skip_features, num_filters, unpool=True):
    if unpool:
        X = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(X)
    X = tf.keras.layers.Concatenate()([X, skip_features])
    X = conv2d_block(X, num_filters)
    return X


def get_g_core_model():
    '''
    mini resnet
    '''
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH]
    inputs = keras.layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
    _, p1 = encoder_block(inputs, 64, pool=False)
    _, p2 = encoder_block(p1, 64)
    _, p3 = encoder_block(p2, 128, pool=False)
    _, p4 = encoder_block(p3, 128)
    _, p5 = encoder_block(p4, 256, pool=False)
    _, p6 = encoder_block(p5, 256)
    _, p7 = encoder_block(p6, 512)
    _, p8 = encoder_block(p7, 512)

    out = tf.keras.layers.GlobalMaxPooling2D()(p8)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation("relu")(out)
    hg = tf.keras.layers.Dense(HIDDEN_SIZE)(out)
    g_net = Model(inputs, hg, name="GNet")
    return g_net


def get_gr_core_model():
    '''
    mini resnet
    '''
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + COORDS + DIST]
    inputs = keras.layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 4))
    _, p1 = encoder_block(inputs, 64, pool=False)
    _, p2 = encoder_block(p1, 64)
    _, p3 = encoder_block(p2, 128, pool=False)
    _, p4 = encoder_block(p3, 128)
    _, p5 = encoder_block(p4, 256, pool=False)
    _, p6 = encoder_block(p5, 256)
    _, p7 = encoder_block(p6, 512)
    _, p8 = encoder_block(p7, 512)
    out = tf.keras.layers.GlobalMaxPooling2D()(p8)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation("relu")(out)
    # z_mean, z_logstd
    z_mean = tf.keras.layers.Dense(HIDDEN_SIZE)(out)
    z_logvar = tf.keras.layers.Dense(HIDDEN_SIZE)(out)
    z = reparameterize(z_mean, z_logvar)
    gr_net = Model(inputs, [z_mean, z_logvar, z], name="GRNet")
    return gr_net


def get_decode_core_model():
    inputs = keras.layers.Input(shape=(HIDDEN_SIZE * 2,))
    R = tf.keras.layers.Dense(HIDDEN_SIZE, use_bias=False)(inputs)
    R = tf.keras.layers.BatchNormalization()(R)
    R = tf.keras.layers.Activation("relu")(R)
    R = tf.keras.layers.Dense(HIDDEN_SIZE * 2, use_bias=False)(R)
    R = tf.keras.layers.BatchNormalization()(R)
    R = tf.keras.layers.Activation("relu")(R)
    R = tf.keras.layers.Dense(MAX_NUM_ATOMS * 3)(R)
    R = tf.reshape(R, [-1, MAX_NUM_ATOMS, 3])
    dec_net = Model(inputs, R, name="Decoder")
    return dec_net
