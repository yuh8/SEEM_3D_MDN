import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from copy import deepcopy
from multiprocessing import freeze_support
from rdkit.Chem.Draw import MolsToGridImage
from src.data_process_utils import connect_3rd_neighbour, mol_to_tensor
from src.bound_utils import embed_conformer, align_conformers
from src.misc_utils import load_json_model, create_folder, pickle_load, pickle_save
from src.CONSTS import TF_EPS, NUM_CONFS_PER_MOL

tfd = tfp.distributions


def loss_func(y_true, y_pred):
    comp_weight, mean, log_std = tf.split(y_pred, 3, axis=-1)
    comp_weight = tf.nn.softmax(comp_weight, axis=-1)
    log_y_true = tf.math.log(y_true + TF_EPS)
    dist = tfd.Normal(loc=mean, scale=tf.math.exp(log_std))
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS, NUM_COMPS]
    _loss = comp_weight * dist.prob(log_y_true)
    # [BATCH, MAX_NUM_ATOMS, MAX_NUM_ATOMS]
    _loss = tf.reduce_sum(_loss, axis=-1)
    _loss = tf.math.log(_loss + TF_EPS)
    mask = tf.squeeze(tf.cast(y_true > 0, tf.float32))
    _loss *= mask
    loss = -tf.reduce_sum(_loss, axis=[1, 2])
    return loss


def plot_3d_scatter(pos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='b', marker='o')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.show()


def get_test_mol(smiles_path):
    mols_origin = []
    mols_pred = []
    drugs_file = "D:/seem_3d_data/data/rdkit_folder/summary_drugs.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    smiles = pickle_load(smiles_path)
    for sim in smiles:
        try:
            mol_path = "D:/seem_3d_data/data/rdkit_folder/" + drugs_summ[sim]['pickle_path']
            with open(mol_path, "rb") as f:
                mol_dict = pickle.load(f)
        except:
            continue

        conf_df = pd.DataFrame(mol_dict['conformers'])
        conf_df.sort_values(by=['boltzmannweight'], ascending=False, inplace=True)
        if conf_df.shape[0] < 1:
            continue
        for _, mol_row in conf_df.iloc[:1, :].iterrows():
            mol_origin = deepcopy(mol_row.rd_mol)
            mol_pred = deepcopy(mol_row.rd_mol)
            g, d = mol_to_tensor(mol_origin, training=False)
            g = connect_3rd_neighbour(mol_origin, g)
            g = np.expand_dims(g, axis=0)
            mask = g.sum(-1) > 3
            d = np.expand_dims(d, axis=0)
            d -= d_mean
            d /= d_std
            # d = np.expand_dims(d, axis=-1)
            d_pred = model(g, training=False).numpy()[0]
            d_pred_mean = np.squeeze(d_pred[..., 1] * mask)
            d_pred_std = np.squeeze(np.exp(d_pred[..., 2]) * mask)

            mol_pred.RemoveAllConformers()
            embed_conformer(mol_pred, d_pred_mean, d_pred_std,
                            d_mean, d_std, np.squeeze(mask), seed=43)
            conf = mol_pred.GetConformers()[0]
            pos = conf.GetPositions()
            plot_3d_scatter(pos)
            align_conformers(mol_pred)
            plot = MolsToGridImage([mol_pred])
            plot.show()
            breakpoint()
            mols_pred.append(mol_pred)
            mols_origin.append(mol_origin)
            if len(mols_pred) % 10 == 0:
                pickle_save(mols_pred, 'generated_confs.pkl')
                pickle_save(mols_origin, 'gt_confs.pkl')


if __name__ == "__main__":
    freeze_support()
    train_path = 'D:/seem_3d_data/train_data/train_batch/'
    test_path = 'D:/seem_3d_data/test_data/test_batch/'

    create_folder('gen_samples/')
    model = load_json_model("conf_model_d_K_1/conf_model_d.json")
    model.compile(optimizer='adam',
                  loss=loss_func)
    model.load_weights("./checkpoints/generator_d_K_1/")
    f_name = train_path + 'stats.pkl'
    with open(f_name, 'rb') as handle:
        d_mean, d_std = pickle.load(handle)

    get_test_mol(test_path + 'smiles.pkl')
