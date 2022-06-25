import tensorflow as tf
from src.layer_utils import Encoder, Sampling
from src.CONSTS import MAX_NUM_ATOMS, NUM_COMPS


class Seem3D(tf.keras.Model):
    def __init__(self, num_filters):
        super().__init__()

        self.enc_0 = Encoder(num_filters, pool=False)
        self.enc_1 = Encoder(num_filters)
        self.enc_2 = Encoder(num_filters * 2, pool=False)
        self.enc_3 = Encoder(num_filters * 2)
        self.enc_4 = Encoder(num_filters * 4, pool=False)
        self.enc_5 = Encoder(num_filters * 4)
        self.enc_6 = Encoder(num_filters * 8, pool=False)
        self.enc_7 = Encoder(num_filters * 8)
        self.global_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.norm = tf.keras.layers.LayerNormalization()
        self.act = tf.keras.layers.Activation("relu")
        self.dense = tf.keras.layers.Dense(MAX_NUM_ATOMS * 6 * NUM_COMPS, use_bias=False)
        self.sample_x = Sampling()
        self.sample_y = Sampling()
        self.sample_z = Sampling()

    def call(self, inputs):
        x = self.enc_0(inputs)
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.enc_4(x)
        x = self.enc_5(x)
        x = self.enc_6(x)
        x = self.enc_7(x)
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dense(x)

        return x

    @tf.function
    def sample(self, inputs, sample_size):
        x = self.enc_0(inputs)
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.enc_4(x)
        x = self.enc_5(x)
        x = self.enc_6(x)
        x = self.enc_7(x)
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dense(x)
        x_mean_std, y_mean_std, z_mean_std = tf.split(x, 3, axis=-1)
        x_coord = self.sample_x(x_mean_std, sample_size)
        y_coord = self.sample_y(y_mean_std, sample_size)
        z_coord = self.sample_z(z_mean_std, sample_size)
        r_pred = tf.concat([x_coord, y_coord, z_coord], axis=-1)

        return r_pred
