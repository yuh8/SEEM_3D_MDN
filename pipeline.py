import os
import tensorflow as tf
from seem_3d_model import Seem3D
from src.misc_utils import align_conf, reparameterize


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


def loss_func(y_true, y_pred):
    # [B,N,1]
    mask = tf.cast(tf.reduce_sum(y_true, axis=-1, keepdims=True) != 0, tf.float32)
    y_pred *= mask
    x_mean_std, y_mean_std, z_mean_std = tf.split(y_pred, 3, axis=-1)

    x_coord = reparameterize(x_mean_std)
    y_coord = reparameterize(y_mean_std)
    z_coord = reparameterize(z_mean_std)
    y_pred_sample = tf.concat([x_coord, y_coord, z_coord], axis=-1)

    y_pred_aligned = tf.stop_gradient(tf.py_function(align_conf,
                                                     inp=[y_pred_sample, y_true, mask],
                                                     Tout=tf.float32))

    total_row = tf.reduce_sum(mask, axis=1, keepdims=True)
    loss = tf.math.squared_difference(y_pred_aligned, y_true)
    loss = tf.reduce_sum(loss, axis=-1)
    loss = tf.reduce_sum(loss, axis=-1) / tf.squeeze(total_row)
    # [BATCH,]
    loss = tf.math.sqrt(loss)
    return loss


class Brain:
    def __init__(self, num_filters):
        self.model = Seem3D(num_filters)
        self.optimizer = get_optimizer()

    def sample(self, G):
        return self.model.sample(G)

    def forward(self, inputs):
        return self.model(inputs)[0]

    @tf.function
    def train_step(self, inputs):
        G = inputs[0]
        R_true = inputs[1]
        with tf.GradientTape() as tape:
            R_pred = self.model(G)
            loss = loss_func(R_true, R_pred)
            # define the negative log-likelihood
            total_loss = tf.reduce_mean(loss)
        model_gradients = tape.gradient(total_loss, self.model.trainable_variables)
        model_gradients, _ = tf.clip_by_global_norm(model_gradients, 0.5)
        self.optimizer.apply_gradients(zip(model_gradients, self.model.trainable_variables))

        return R_pred, loss

    @tf.function
    def val_step(self, inputs):
        G = inputs[0]
        R_true = inputs[1]
        R_pred = self.model(G)
        loss = loss_func(R_true, R_pred)

        return R_pred, loss

    def save_weights(self, path):
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        # Save the weights
        self.model.save_weights(path + ".h5")

    def load_weights(self, path):
        try:
            self.model.load_weights(path + ".h5")
        except:
            return "Weights cannot be loaded"
        return "Weights loaded"
