import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers


class BottleNeck(layers.Layer):
    def __init__(self, num_filters, kernel_size=3, padding='SAME'):
        super(BottleNeck, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding

    def build(self, input_shape):
        self.bottle_0 = tf.keras.layers.Conv2D(self.num_filters // 4,
                                               kernel_size=1,
                                               padding=self.padding)
        self.bottle_1 = tf.keras.layers.Conv2D(self.num_filters // 4,
                                               kernel_size=self.kernel_size,
                                               padding=self.padding)
        self.bottle_2 = tf.keras.layers.Conv2D(self.num_filters,
                                               kernel_size=1,
                                               padding=self.padding)

    def call(self, inputs):
        x = self.bottle_0(inputs)
        x = self.bottle_1(x)
        x = self.bottle_2(x)
        return x


class Conv2DBlock(layers.Layer):
    def __init__(self, num_filters, kernel_size=3, padding="SAME"):
        super(Conv2DBlock, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding

    def build(self, input_shape):
        self.bottle = BottleNeck(self.num_filters)
        self.conv_2d = tf.keras.layers.Conv2D(self.num_filters,
                                              kernel_size=self.kernel_size,
                                              padding=self.padding)
        self.norm = tf.keras.layers.LayerNormalization()
        self.act = tf.keras.layers.Activation("relu")

    def call(self, inputs):
        if self.num_filters >= 256:
            x = self.bottle(inputs)
        else:
            x = self.conv_2d(inputs)
        x = self.norm(x)
        x = self.act(x)
        return x


class Encoder(layers.Layer):
    def __init__(self, num_filters, kernel_size=3, padding="SAME", pool=True):
        super(Encoder, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.pool = pool

    def build(self, input_shape):
        self.conv_block_0 = Conv2DBlock(self.num_filters)
        self.conv_block_1 = Conv2DBlock(self.num_filters)
        self.conv_block_2 = Conv2DBlock(self.num_filters)
        self.pool = tf.keras.layers.MaxPool2D(2, 2)

    def call(self, inputs):
        x = self.conv_block_0(inputs)
        x = self.conv_block_1(inputs)
        x = self.conv_block_2(inputs)
        if self.pool:
            x = tf.keras.layers.MaxPool2D(2, 2)(x)
        return x


class Sampling(layers.Layer):

    def call(self, inputs, sample_size=100):
        z_mean = tf.expand_dims(inputs[..., 0], axis=-1)
        z_log_var = tf.expand_dims(inputs[..., 1], axis=-1)
        batch = sample_size
        dim_0 = tf.shape(z_mean)[1]
        dim_1 = tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim_0, dim_1))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
