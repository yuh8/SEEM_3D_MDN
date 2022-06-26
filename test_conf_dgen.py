import json
import pickle
from random import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from copy import deepcopy
from multiprocessing import freeze_support
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem.rdMolAlign import GetBestRMS
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMoleculeConfs
from src.data_process_utils import mol_to_tensor
from src.bound_utils import embed_conformer
from src.misc_utils import load_json_model, create_folder, pickle_load, pickle_save
from src.CONSTS import TF_EPS

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


def get_best_RMSD(probe, ref, prbid, refid=-1):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)
    rmsd = GetBestRMS(probe, ref, prbid, refid)
    return rmsd


def get_prediction(mol):
    mol_origin = deepcopy(mol)
    g, _, R = mol_to_tensor(mol_origin)
    g = np.expand_dims(g, axis=0)
    mask = np.abs(R).sum(-1, keepdims=True) > 0
    d_pred_mean = model(g, training=False).numpy()[0][..., [0, 2, 4]] * mask
    return d_pred_mean


def compute_cov_mat(smiles_path):
    drugs_file = "D:/seem_3d_data/data/rdkit_folder/summary_drugs.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    smiles = pickle_load(smiles_path)
    shuffle(smiles)
    covs = []
    mats = []
    for smi in smiles[:200]:
        try:
            mol_path = "D:/seem_3d_data/data/rdkit_folder/" + drugs_summ[smi]['pickle_path']
            with open(mol_path, "rb") as f:
                mol_dict = pickle.load(f)
        except:
            continue

        conf_df = pd.DataFrame(mol_dict['conformers'])
        conf_df.sort_values(by=['boltzmannweight'], ascending=False, inplace=True)

        if conf_df.shape[0] < 1:
            continue

        mol_pred = deepcopy(conf_df.iloc[0].rd_mol)

        try:
            d_pred_mean = get_prediction(mol_pred)
        except:
            continue
        num_refs = conf_df.shape[0]
        num_gens = num_refs * 2

        embed_conformer(mol_pred, num_gens, d_pred_mean, d_pred_std,
                        d_mean, d_std, np.squeeze(mask), seed=43)

        num_gens = mol_pred.GetNumConformers()
        cov_mat = np.zeros((num_refs, num_gens))
        # MMFFOptimizeMoleculeConfs(mol_pred)

        cnt = 0
        try:
            for _, mol_row in conf_df.iterrows():
                mol_prob = deepcopy(mol_pred)
                mol_ref = deepcopy(mol_row.rd_mol)
                for j in range(num_gens):
                    rmsd = get_best_RMSD(mol_prob, mol_ref, j)
                    cov_mat[cnt, j] = rmsd
                cnt += 1
        except:
            continue
        cov_score = np.mean(cov_mat.min(-1) < 1.25)
        mat_score = np.sum(cov_mat.min(-1)) / num_refs
        print('cov_score and mat_score for {0} is {1} and {2}'.format(smi, cov_score, mat_score))
        covs.append(cov_score)
        mats.append(mat_score)
    breakpoint()


if __name__ == "__main__":
    freeze_support()
    train_path = 'D:/seem_3d_data/train_data/train_batch/'
    test_path = 'D:/seem_3d_data/test_data/test_batch/'

    create_folder('gen_samples/')
    model = load_json_model("conf_model_R_K_1/conf_model_d.json")
    model.compile(optimizer='adam',
                  loss=loss_func)
    model.load_weights("./checkpoints/generator_R_K_1/")
    f_name = train_path + 'stats.pkl'
    with open(f_name, 'rb') as handle:
        d_mean, d_std = pickle.load(handle)

    compute_cov_mat(test_path + 'smiles.pkl')
