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


def conv2d_block(input, num_filters, kernel_size=3, padding='SAME'):
    x = tf.keras.layers.Conv2D(num_filters,
                               kernel_size=kernel_size,
                               padding=padding)(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

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
