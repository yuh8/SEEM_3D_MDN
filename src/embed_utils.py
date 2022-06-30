import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

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

    if num_filters >= 256:
        x = bottleneck(x, num_filters)
    else:
        x = tf.keras.layers.Conv2D(num_filters,
                                   kernel_size=kernel_size,
                                   padding=padding)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
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
