import glob
import pickle
import numpy as np
import tensorflow as tf
from datetime import date
from multiprocessing import freeze_support
from pipeline import Brain
from src.misc_utils import create_folder
from src.CONSTS import NUM_COMPS


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
train_log_dir = 'logs_{}/'.format(today)
writer = tf.summary.create_file_writer(train_log_dir)
tf.summary.trace_on(graph=True, profiler=False)

train_rmsd = tf.keras.metrics.Mean("train_rmsd")


def data_iterator(data_path):
    num_files = len(glob.glob(data_path + 'GDR_*.pkl'))
    batch_nums = np.arange(num_files)
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


if __name__ == "__main__":
    freeze_support()
    ckpt_path = 'checkpoints/generator_R_K_{}/'.format(NUM_COMPS)
    create_folder(ckpt_path)
    create_folder("conf_model_R_K_{}".format(NUM_COMPS))
    train_path = 'D:/seem_3d_data/train_data/train_batch/'
    val_path = 'D:/seem_3d_data/test_data/val_batch/'
    test_path = 'D:/seem_3d_data/test_data/test_batch/'

    brain = Brain(64)

    with writer.as_default():
        for epoch in range(2):
            train_rmsd.reset_states()
            train_step = 0
            val_rmsd = 0
            for g, r in data_iterator(train_path):
                if train_step == 0:
                    tf.summary.trace_export(name="seem_3d", step=train_step)
                R_pred, _loss, = brain.train_step([g, r])
                train_rmsd.update_state(_loss)

                tf.summary.scalar('train_rmsd', train_rmsd.result(), step=train_step)
                train_step += 1
                if train_step % 10 == 0:
                    print("train_rmsd = {0} at step {1}".format(np.mean(_loss, train_step)))
                    writer.flush()
                if train_step % 1000 == 0:
                    brain.save_weights(ckpt_path)

            val_step = 0
            for g, r in data_iterator(val_path):
                R_pred, val_loss = brain.val_step([g, r])
                val_rmsd += val_loss.numpy()
                val_step += 1

            val_rmsd = np.round(val_rmsd / val_step, 3)
            tf.summary.scalar('val_rmsd', val_rmsd, step=epoch)
            writer.flush()
