import glob
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from datetime import date
from tensorflow import keras
from tensorflow.keras import models, Model
from multiprocessing import freeze_support
from src.embed_utils import get_dist_core_model, get_r_core_model
from src.misc_utils import create_folder, save_model_to_json, align_conf, tf_contriod
from src.CONSTS import NUM_COMPS, TF_EPS, MAX_NUM_ATOMS, FEATURE_DEPTH

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


def reparameterize(mean, logstd):
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(mean))
    return mean + tf.exp(logstd) * epsilon


def sample_dist(y_pred):
    _, mean, log_std = tf.split(y_pred, 3, axis=-1)
    dist_sample = reparameterize(mean, log_std)
    dist_sample = tf.squeeze(dist_sample)
    diag = tf.zeros((tf.shape(dist_sample)[0],
                     tf.shape(dist_sample)[1]))
    dist_sample = tf.linalg.set_diag(dist_sample, diag)
    dist_sample = tf.expand_dims(dist_sample, axis=-1)
    return dist_sample


def core_model():
    inputs = keras.layers.Input(shape=(MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
    dist_pred = dist_net(inputs)
    dist_sample = sample_dist(dist_pred)
    r_pred = r_net(dist_sample)
    return inputs, dist_pred, r_pred


def loss_func_dist(y_true, y_pred):
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


def loss_func_r(y_true, y_pred):
    # [B,N,1]
    mask = tf.cast(tf.reduce_sum(tf.abs(y_true), axis=-1, keepdims=True) != 0, tf.float32)
    y_pred *= mask
    Rot = tf.stop_gradient(tf.py_function(align_conf,
                                          inp=[y_pred, y_true, mask],
                                          Tout=tf.float32))
    QC = tf.stop_gradient(tf_contriod(y_true, mask))
    y_pred_aligned = tf.matmul(y_pred, Rot) + QC
    y_pred_aligned *= mask
    total_row = tf.reduce_sum(mask, axis=1, keepdims=True)
    loss = tf.math.squared_difference(y_pred_aligned, y_true)
    loss = tf.reduce_sum(loss, axis=-1)
    loss = tf.reduce_sum(loss, axis=-1) / tf.squeeze(total_row)
    # [BATCH,]
    loss = tf.math.sqrt(loss)
    return loss


def loss_func(y_true, y_pred):
    dist_true, r_true = y_true
    dist_pred, r_pred = y_pred
    loss_dist = loss_func_dist(dist_true, dist_pred)
    loss_r = loss_func_r(r_true, r_pred)
    loss = loss_dist + 0.01 * loss_r
    return loss


def distance_rmse(d_true, d_pred):
    comp_weight, mean, _ = tf.split(d_pred, 3, axis=-1)
    comp_weight = tf.nn.softmax(comp_weight, axis=-1)
    se = (mean - d_true)**2
    se *= comp_weight
    se = tf.reduce_sum(se, axis=-1)
    mask = tf.squeeze(tf.cast(d_true != 0, tf.float32))
    se *= mask
    loss = tf.reduce_sum(se, axis=[1, 2]) / tf.reduce_sum(mask, axis=[1, 2])
    loss = tf.math.sqrt(loss)
    return loss


def distance_rmsd(r_true, r_pred):
    # [B,N,1]
    mask = tf.cast(tf.reduce_sum(tf.abs(r_true), axis=-1, keepdims=True) != 0, tf.float32)
    r_pred *= mask
    Rot = tf.stop_gradient(tf.py_function(align_conf,
                                          inp=[r_pred, r_true, mask],
                                          Tout=tf.float32))
    QC = tf.stop_gradient(tf_contriod(r_true, mask))
    y_pred_aligned = tf.matmul(r_pred, Rot) + QC
    y_pred_aligned *= mask
    total_row = tf.reduce_sum(mask, axis=1, keepdims=True)
    loss = tf.math.squared_difference(y_pred_aligned, r_true)
    loss = tf.reduce_sum(loss, axis=-1)
    loss = tf.reduce_sum(loss, axis=-1) / tf.squeeze(total_row)
    # [BATCH,]
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
        [1000000, 2000000, 3000000], [lr, lr / 10, lr / 50, lr / 100],
        name=None
    )
    opt_op = tf.keras.optimizers.Adam(learning_rate=lr_fn, global_clipnorm=0.5)
    return opt_op


def get_metrics():
    dist_logden = tf.keras.metrics.Mean(name="dist_logden")
    dist_rmse = tf.keras.metrics.Mean(name="dist_rmse")
    r_rmsd = tf.keras.metrics.Mean(name="dist_rmse")
    return dist_logden, dist_rmse, r_rmsd


class Seem3D(Model):
    def compile(self, optimizer, loss_fn, metrics):
        super(Seem3D, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dist_logden = metrics[0]
        self.dist_rmse = metrics[1]
        self.r_rmsd = metrics[2]

    def train_step(self, data):
        X = data[0]
        dist_true = data[1][0]
        r_true = data[1][1]
        y_true = (dist_true, r_true)

        # capture the scope of gradient
        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)
            loss = self.loss_fn(y_true, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.dist_logden.update_state(loss)
        self.dist_rmse.update_state(distance_rmse(y_true[0], y_pred[0]))
        self.r_rmsd.update_state(distance_rmsd(y_true[1], y_pred[1]))
        return {"dist_logden": self.dist_logden.result(),
                "dist_rmse": self.dist_rmse.result(),
                "r_rmsd": self.r_rmsd.result()}

    def test_step(self, data):
        X = data[0]
        dist_true = data[1][0]
        r_true = data[1][1]
        y_true = (dist_true, r_true)

        # capture the scope of gradient
        y_pred = self(X, training=True)
        loss = self.loss_fn(y_true, y_pred)
        self.dist_logden.update_state(loss)
        self.dist_rmse.update_state(distance_rmse(y_true[0], y_pred[0]))
        self.r_rmsd.update_state(distance_rmsd(y_true[1], y_pred[1]))
        return {"dist_logden": self.dist_logden.result(),
                "dist_rmse": self.dist_rmse.result(),
                "r_rmsd": self.r_rmsd.result()}

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.dist_logden, self.dist_rmse, self.r_rmsd]


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
            R = GD[2].todense()
            D -= d_mean
            D /= d_std
            mask = G.sum(-1) > 3
            D *= mask

            sample_nums = np.arange(G.shape[0])
            np.random.shuffle(sample_nums)
            yield G[sample_nums, ...], (np.expand_dims(D[sample_nums, ...], axis=-1),
                                        R[sample_nums, ...])


def data_iterator_test(data_path):
    num_files = len(glob.glob(data_path + 'GDR_*.pkl'))
    batch_nums = np.arange(num_files)
    for batch in batch_nums:
        f_name = data_path + 'GDR_{}.pkl'.format(batch)
        with open(f_name, 'rb') as handle:
            GD = pickle.load(handle)

        G = GD[0].todense()
        D = GD[1].todense()
        R = GD[2].todense()
        D -= d_mean
        D /= d_std
        mask = G.sum(-1) > 3
        D *= mask
        yield G, (np.expand_dims(D, axis=-1), R)


if __name__ == "__main__":
    freeze_support()
    ckpt_path = 'checkpoints/generator_d_K_{}/'.format(NUM_COMPS)
    create_folder(ckpt_path)
    create_folder("conf_model_d_K_{}".format(NUM_COMPS))
    create_folder("dist_net")
    create_folder("r_net")
    train_path = 'D:/seem_3d_data/train_data/train_batch/'
    val_path = 'D:/seem_3d_data/test_data/val_batch/'
    test_path = 'D:/seem_3d_data/test_data/test_batch/'

    f_name = train_path + 'stats.pkl'
    with open(f_name, 'rb') as handle:
        d_mean, d_std = pickle.load(handle)

    train_steps = len(glob.glob(train_path + 'GDR_*.pkl'))
    val_steps = len(glob.glob(val_path + 'GDR_*.pkl'))

    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                    save_freq=1000,
                                                    save_weights_only=True,
                                                    monitor='dist_logden',
                                                    mode='min',
                                                    save_best_only=True)]
    dist_net = get_dist_core_model()
    r_net = get_r_core_model()

    X, dist_pred, r_pred = core_model()

    model = Seem3D(inputs=X, outputs=[dist_pred, r_pred])

    model.compile(optimizer=get_optimizer(),
                  loss_fn=loss_func, metrics=get_metrics())

    save_model_to_json(model, "conf_model_d_K_{}/conf_model_d.json".format(NUM_COMPS))
    model.summary()

    try:
        model.load_weights("./checkpoints/generator_d_K_1/")
    except:
        print('no exitsing model detected, training starts afresh')
        pass

    model.fit(data_iterator(train_path),
              epochs=40,
              validation_data=data_iterator(val_path),
              validation_steps=val_steps,
              callbacks=callbacks,
              steps_per_epoch=train_steps)
    res = model.evaluate(data_iterator_test(test_path),
                         return_dict=True)

    # save trained model in two ways
    dist_net.save('dist_net/' + 'DistNet')
    r_net.save('r_net/' + 'RNet')
