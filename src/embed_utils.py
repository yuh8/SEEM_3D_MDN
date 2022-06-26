import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from .CONSTS import (MAX_NUM_ATOMS, FEATURE_DEPTH, OUTPUT_DEPTH, NUM_COMPS)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


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
    x = tf.keras.layers.LayerNormalization()(x)

    # if num_filters >= 256:
    #     x = bottleneck(x, num_filters)
    # else:
    #     x = tf.keras.layers.Conv2D(num_filters,
    #                                kernel_size=kernel_size,
    #                                padding=padding)(x)
    # x = tf.keras.layers.Activation("relu")(x)
    # x = tf.keras.layers.LayerNormalization()(x)
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


def get_dist_core_model():
    '''
    mini unet
    '''
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH]
    inputs = keras.layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
    s1, p1 = encoder_block(inputs, 64, pool=False)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128, pool=False)
    s4, p4 = encoder_block(p3, 128)
    s5, p5 = encoder_block(p4, 256, pool=False)
    s6, p6 = encoder_block(p5, 384)

    b1 = conv2d_block(p6, 512)

    d1 = decoder_block(b1, s6, 384)
    d2 = decoder_block(d1, s5, 256, unpool=False)
    d3 = decoder_block(d2, s4, 128)
    d4 = decoder_block(d3, s3, 128, unpool=False)
    d5 = decoder_block(d4, s2, 64)
    d6 = decoder_block(d5, s1, 64, unpool=False)

    # Add a per-pixel classification layer
    y_pred = keras.layers.Conv2D(OUTPUT_DEPTH, 1, activation=None, padding="same", use_bias=False)(d6)
    dist_net = Model(inputs, y_pred, name="distance_net")
    return dist_net


def get_r_core_model():
    '''
    mini resnet
    '''
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH]
    inputs = keras.layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, 1))
    _, p1 = encoder_block(inputs, 32)
    _, p2 = encoder_block(p1, 32)
    _, p3 = encoder_block(p2, 64)
    _, p4 = encoder_block(p3, 64)
    _, p5 = encoder_block(p4, 128)

    out = tf.keras.layers.GlobalMaxPooling2D()(p5)
    out = tf.keras.layers.LayerNormalization()(out)
    out = tf.keras.layers.Activation("relu")(out)

    out = tf.keras.layers.Dense(MAX_NUM_ATOMS * 3, use_bias=False)(out)
    out = tf.reshape(out, [-1, MAX_NUM_ATOMS, 3])
    r_net = Model(inputs, out, name="position_net")
    return r_net
