import glob
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from multiprocessing import freeze_support
from src.embed_utils import get_g_core_model, get_gr_core_model, get_decode_core_model
from src.misc_utils import create_folder, align_conf, tf_contriod
from src.CONSTS import MAX_NUM_ATOMS, FEATURE_DEPTH

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=1000)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def get_mertcis():
    kl = keras.metrics.Mean(name="kl_loss")
    r_rmsd = keras.metrics.Mean(name="r_rmsd")
    return kl, r_rmsd


def core_model():
    inputs = keras.layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 4))
    z_mean, z_log_var, z = gr_net(inputs)
    h = g_net(inputs[..., :-4])
    hz = tf.concat([h, z], axis=-1)
    r_pred = dec_net(hz)
    return inputs, z_mean, z_log_var, r_pred


def loss_func_r(y_true, y_pred):
    # [B,N,1]
    mask = tf.cast(tf.reduce_sum(tf.abs(y_true), axis=-1, keepdims=True) > 0, tf.float32)
    y_pred *= mask
    Rot = tf.stop_gradient(tf.py_function(align_conf,
                                          inp=[y_pred, y_true, mask],
                                          Tout=tf.float32))
    QC = tf_contriod(y_true, mask)
    y_pred_aligned = tf.matmul(y_pred, Rot) + QC
    y_pred_aligned *= mask
    total_row = tf.reduce_sum(mask, axis=1, keepdims=True)
    loss = tf.math.squared_difference(y_pred_aligned, y_true)
    loss = tf.reduce_sum(loss, axis=-1)
    loss = tf.reduce_sum(loss, axis=-1) / tf.squeeze(total_row)
    # [BATCH,]
    loss = tf.math.sqrt(loss)
    return loss


def loss_func_kl(z_mean, z_logvar):
    kl_loss = -0.5 * (1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
    kl_loss = tf.reduce_sum(kl_loss, axis=1)
    return kl_loss


def get_optimizer(finetune=False):
    lr = 0.0002
    if finetune:
        lr = 0.00001
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [240000, 480000], [lr, lr / 10, lr / 50],
        name=None
    )
    opt_op = tf.keras.optimizers.Adam(learning_rate=lr_fn, global_clipnorm=0.5)
    return opt_op


def get_metrics():
    kl = tf.keras.metrics.Mean(name="kl_loss")
    r_rmsd = tf.keras.metrics.Mean(name="r_rmsd")
    return kl, r_rmsd


class Seem3D(Model):
    def compile(self, optimizer, metrics):
        super(Seem3D, self).compile()
        self.optimizer = optimizer
        self.kl = metrics[0]
        self.r_rmsd = metrics[1]

    def train_step(self, data):
        X = data[0]
        r_true = data[1]

        # capture the scope of gradient
        with tf.GradientTape() as tape:
            z_mean, z_log_var, r_pred = self(X, training=True)
            kl_loss = loss_func_kl(z_mean, z_log_var)
            rec_loss = loss_func_r(r_true, r_pred)
            loss = 0.001 * kl_loss + rec_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.kl.update_state(kl_loss)
        self.r_rmsd.update_state(rec_loss)
        return {"kl_loss": self.kl.result(),
                "r_rmsd": self.r_rmsd.result()}

    def test_step(self, data):
        X = data[0]
        r_true = data[1]
        z_mean, z_log_var, r_pred = self(X, training=False)
        kl_loss = loss_func_kl(z_mean, z_log_var)
        rec_loss = loss_func_r(r_true, r_pred)
        self.kl.update_state(kl_loss)
        self.r_rmsd.update_state(rec_loss)
        return {"kl_loss": self.kl.result(),
                "r_rmsd": self.r_rmsd.result()}

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.kl, self.r_rmsd]


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
            R = GD[1].todense()

            sample_nums = np.arange(G.shape[0])
            np.random.shuffle(sample_nums)
            yield G[sample_nums, ...], R[sample_nums, ...]


def data_iterator_test(data_path):
    num_files = len(glob.glob(data_path + 'GDR_*.pkl'))
    batch_nums = np.arange(num_files)
    for batch in batch_nums:
        f_name = data_path + 'GDR_{}.pkl'.format(batch)
        with open(f_name, 'rb') as handle:
            GD = pickle.load(handle)

        G = GD[0].todense()
        R = GD[1].todense()
        yield G, R


if __name__ == "__main__":
    freeze_support()
    ckpt_path = 'checkpoints/conVAE/'
    create_folder(ckpt_path)
    create_folder("conVAE")
    create_folder("dec_net")
    create_folder("gr_net")
    create_folder("g_net")
    train_path = 'D:/seem_3d_data/train_data/train_batch/'
    val_path = 'D:/seem_3d_data/test_data/val_batch/'
    test_path = 'D:/seem_3d_data/test_data/test_batch/'

    train_steps = len(glob.glob(train_path + 'GDR_*.pkl'))
    val_steps = len(glob.glob(val_path + 'GDR_*.pkl'))

    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                    save_freq=1000,
                                                    save_weights_only=True,
                                                    monitor='r_rmsd',
                                                    mode='min',
                                                    save_best_only=True)]
    g_net = get_g_core_model()
    gr_net = get_gr_core_model()
    dec_net = get_decode_core_model()

    X, z_mean, z_log_var, r_pred = core_model()
    convae = Seem3D(inputs=X, outputs=[z_mean, z_log_var, r_pred])
    optimizer = get_optimizer()
    convae.compile(optimizer=get_optimizer(), metrics=get_metrics())
    convae.summary()
    breakpoint()

    try:
        convae.load_weights("./checkpoints/conVAE/")
    except:
        print('no exitsing model detected, training starts afresh')
        pass

    convae.fit(data_iterator(train_path),
               epochs=100,
               validation_data=data_iterator(val_path),
               validation_steps=val_steps,
               callbacks=callbacks,
               steps_per_epoch=train_steps)
    res = convae.evaluate(data_iterator_test(test_path),
                          return_dict=True)

    # save trained model in two ways
    g_net.save('g_net/' + 'GNet')
    gr_net.save('gr_net/' + 'GRNet')
    dec_net.save('dec_net/' + 'DecNet')
