import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from multiprocessing import freeze_support
from src.embed_utils import get_g_net, get_gdr_net, get_decode_net
from src.misc_utils import create_folder, align_conf, tf_contriod
from src.CONSTS import MAX_NUM_ATOMS, FEATURE_DEPTH, BATCH_SIZE, VAL_BATCH_SIZE

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
    mask = tf.reduce_sum(tf.abs(inputs), axis=-1)
    mask = tf.reduce_sum(mask, axis=1, keepdims=True) <= 0
    mask = tf.cast(mask, tf.float32)
    z_mean, z_log_var, z = gdr_net(inputs)
    h = g_net(inputs[..., :-4])
    r_pred = dec_net([h, mask, z])
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
    # [batch_size, num_atoms, d_model]
    kl_loss = -0.5 * (1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
    kl_loss = tf.reduce_sum(kl_loss, axis=[1, 2])
    return kl_loss


class WarmCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps=4000):
        super(WarmCosine, self).__init__()
        self.d_model = 9216
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def get_optimizer():
    opt_op = tf.keras.optimizers.Adam(learning_rate=WarmCosine(), clipnorm=0.5)
    return opt_op


def get_metrics():
    kl = tf.keras.metrics.Mean(name="kl_loss")
    r_rmsd = tf.keras.metrics.Mean(name="r_rmsd")
    return kl, r_rmsd


class TransVAE(Model):
    def compile(self, optimizer, metrics):
        super(TransVAE, self).compile()
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
            loss = 0.01 * kl_loss + rec_loss

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


def data_iterator_train():
    num_files = len(glob.glob(train_path + 'GDR_*.pkl'))
    batch_nums = np.arange(num_files)
    while True:
        np.random.shuffle(batch_nums)
        for batch in batch_nums:
            f_name = train_path + f'GDR_{batch}.npz'
            GDR = np.load(f_name)
            G = GDR['G']
            R = GDR['R']
            yield G, R


def data_iterator_val():
    num_files = len(glob.glob(val_path + 'GDR_*.pkl'))
    batch_nums = np.arange(num_files)
    while True:
        np.random.shuffle(batch_nums)
        for batch in batch_nums:
            f_name = val_path + f'GDR_{batch}.npz'
            GDR = np.load(f_name)
            G = GDR['G']
            R = GDR['R']
            yield G, R


def data_iterator_test():
    num_files = len(glob.glob(test_path + 'GDR_*.pkl'))
    batch_nums = np.arange(num_files)
    for batch in batch_nums:
        f_name = test_path + f'GDR_{batch}.npz'
        GDR = np.load(f_name)
        G = GDR['G']
        R = GDR['R']
        yield G, R


def _fixup_shape(x, y):
    x.set_shape([None, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 4])
    y.set_shape([None, MAX_NUM_ATOMS, 3])
    return x, y


if __name__ == "__main__":
    freeze_support()
    ckpt_path = 'checkpoints/TransVAE/'
    create_folder(ckpt_path)
    create_folder("TransVAE")
    create_folder("dec_net")
    create_folder("gdr_net")
    create_folder("g_net")
    train_path = '/mnt/transvae/train_data/train_batch/'
    val_path = '/mnt/transvae/test_data/val_batch/'
    test_path = '/mnt/transvae/test_data/test_batch/'

    train_steps = len(glob.glob(train_path + 'GDR_*.pkl')) // BATCH_SIZE
    val_steps = len(glob.glob(val_path + 'GDR_*.pkl')) // VAL_BATCH_SIZE

    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                    save_freq=1000,
                                                    save_weights_only=True,
                                                    monitor='r_rmsd',
                                                    mode='min',
                                                    save_best_only=True)]
    g_net = get_g_net()
    gdr_net = get_gdr_net()
    dec_net = get_decode_net()

    X, z_mean, z_log_var, r_pred = core_model()
    transvae = TransVAE(inputs=X, outputs=[z_mean, z_log_var, r_pred])
    optimizer = get_optimizer()
    transvae.compile(optimizer=get_optimizer(), metrics=get_metrics())
    transvae.summary()
    breakpoint()

    try:
        transvae.load_weights("./checkpoints/TransVAE/")
    except:
        print('no exitsing model detected, training starts afresh')
        pass

    train_dataset = tf.data.Dataset.from_generator(
        data_iterator_train,
        output_types=(tf.float32, tf.float32),
        output_shapes=((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 4),
                       (MAX_NUM_ATOMS, 3)))

    train_dataset = train_dataset.shuffle(buffer_size=1000, seed=0,
                                          reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True).map(_fixup_shape)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        data_iterator_val,
        output_types=(tf.float32, tf.float32),
        output_shapes=((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 4),
                       (MAX_NUM_ATOMS, 3)))
    val_dataset = val_dataset.batch(VAL_BATCH_SIZE, drop_remainder=True).map(_fixup_shape)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_generator(
        data_iterator_test,
        output_types=(tf.float32, tf.float32),
        output_shapes=((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 4),
                       (MAX_NUM_ATOMS, 3)))
    test_dataset = test_dataset.batch(VAL_BATCH_SIZE, drop_remainder=True).map(_fixup_shape)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    transvae.fit(train_dataset,
                 epochs=50,
                 validation_data=val_dataset,
                 validation_steps=val_steps,
                 callbacks=callbacks,
                 steps_per_epoch=train_steps)
    res = transvae.evaluate(test_dataset,
                            return_dict=True)

    # save trained model in two ways
    g_net.compile(optimizer='adam', loss=None)
    g_net.save('g_net/' + 'GNet')
    gdr_net.compile(optimizer='adam', loss=None)
    gdr_net.save('gr_net/' + 'GDRNet')
    dec_net.compile(optimizer='adam', loss=None)
    dec_net.save('dec_net/' + 'DecNet')
