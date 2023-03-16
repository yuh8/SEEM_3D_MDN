import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import pickle
from random import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from copy import deepcopy
from multiprocessing import freeze_support
from rdkit.Geometry import Point3D
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem.rdMolAlign import GetBestRMS
from src.data_process_utils import mol_to_tensor
from src.misc_utils import pickle_load
from src.CONSTS import HIDDEN_SIZE, MAX_NUM_ATOMS, TF_EPS


def load_models():
    g_net = tf.keras.models.load_model('g_net_qm9/GNet/')
    gr_net = tf.keras.models.load_model('gr_net_qm9/GDRNet/')
    decoder_net = tf.keras.models.load_model('dec_net_qm9/DecNet/')
    return g_net, decoder_net, gr_net


def plot_3d_scatter(pos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='b', marker='o')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.savefig('./conf_1.png')


def get_best_RMSD(probe, ref, prbid=-1, refid=-1):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)
    rmsd = GetBestRMS(probe, ref, prbid, refid)
    return rmsd


def get_prediction(mol, sample_size, g_net, decoder_net):
    mol_origin = deepcopy(mol)
    gr, _ = mol_to_tensor(mol_origin)
    g = np.expand_dims(gr, axis=0)[..., :-4]
    mask = np.sum(np.abs(g), axis=-1)
    mask = np.sum(mask, axis=1, keepdims=True) <= 0
    mask = np.expand_dims(mask, axis=1).astype(np.float32)
    with tf.device('/cpu:2'):
        h = g_net.predict(g)
    h = np.tile(h, [sample_size, 1, 1])
    mask = np.tile(mask, [sample_size, 1, 1, 1])
    z = np.random.normal(0, 1, size=(sample_size, MAX_NUM_ATOMS, HIDDEN_SIZE))
    with tf.device('/cpu:2'):
        r_pred = decoder_net.predict([h, mask, z]) * (1 - np.squeeze(mask)[..., np.newaxis])
    return r_pred


def get_mol_probs(mol_pred, r_pred, num_gens, FF=True):
    mol_probs = []
    for j in range(num_gens):
        mol_prob = deepcopy(mol_pred)
        _conf = mol_prob.GetConformer()
        for i in range(mol_prob.GetNumAtoms()):
            _conf.SetAtomPosition(i, r_pred[j][i].tolist())
        if FF:
            MMFFOptimizeMolecule(mol_prob)
        mol_probs.append(mol_prob)
    return mol_probs


def compute_cov_mat(smiles_path, g_net, decoder_net):
    drugs_file = "/mnt/raw_data/rdkit_folder/summary_qm9.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    smiles = pickle_load(smiles_path)
    # shuffle(smiles)
    breakpoint()

    cov_means = []
    cov_meds = []
    mat_means = []
    mat_meds = []

    covs = []
    mats = []
    for idx, smi in enumerate(smiles):
        try:
            mol_path = "/mnt/raw_data/rdkit_folder/" + drugs_summ[smi]['pickle_path']
            with open(mol_path, "rb") as f:
                mol_dict = pickle.load(f)
        except:
            print('smiles missing')
            continue

        conf_df = pd.DataFrame(mol_dict['conformers'])
        conf_df.sort_values(by=['boltzmannweight'], ascending=False, inplace=True)

        num_refs = conf_df.shape[0]

        if num_refs < 50:
            continue

        if num_refs > 500:
            continue

        num_gens = num_refs * 2

        mol_pred = deepcopy(conf_df.iloc[0].rd_mol)

        try:
            r_pred = get_prediction(mol_pred, num_gens, g_net, decoder_net)
        except:
            continue

        cov_mat = np.zeros((conf_df.shape[0], num_gens))

        cnt = 0
        mol_probs = get_mol_probs(mol_pred, r_pred, num_gens, FF=False)
        try:
            for _, mol_row in conf_df.iterrows():
                mol_ref = deepcopy(mol_row.rd_mol)
                for j in range(num_gens):
                    rmsd = get_best_RMSD(mol_probs[j], mol_ref)
                    cov_mat[cnt, j] = rmsd
                cnt += 1
        except:
            continue

        cov_score = np.mean(cov_mat.min(-1) < 0.5)
        mat_score = np.sum(cov_mat.min(-1)) / conf_df.shape[0]
        covs.append(cov_score)
        mats.append(mat_score)
        cov_mean = np.round(np.mean(covs), 4)
        cov_med = np.round(np.median(covs), 4)
        mat_mean = np.round(np.mean(mats), 4)
        mat_med = np.round(np.median(mats), 4)
        print(f'cov_mean = {cov_mean}, cov_med = {cov_med}, mat_mean = {mat_mean}, mat_med = {mat_med} for {idx} th mol')
        if len(covs) == 200:
            cov_means.append(cov_mean)
            cov_meds.append(cov_med)
            mat_means.append(mat_mean)
            mat_meds.append(mat_med)
            covs = []
            mats = []

        if len(cov_means) == 10:
            break

    cov_means_mean = np.round(np.mean(cov_means), 4)
    cov_means_std = np.round(np.std(cov_means), 4)
    cov_meds_mean = np.round(np.mean(cov_meds), 4)
    cov_meds_std = np.round(np.std(cov_meds), 4)

    mat_means_mean = np.round(np.mean(mat_means), 4)
    mat_means_std = np.round(np.std(mat_means), 4)
    mat_meds_mean = np.round(np.mean(mat_meds), 4)
    mat_meds_std = np.round(np.std(mat_meds), 4)
    print(f'cov_means_mean = {cov_means_mean} with std {cov_means_std}')
    print(f'cov_meds_mean = {cov_meds_mean} with std {cov_meds_std}')
    print(f'mat_means_mean = {mat_means_mean} with std {mat_means_std}')
    print(f'mat_meds_mean = {mat_meds_mean} with std {mat_meds_std}')


if __name__ == "__main__":
    freeze_support()
    g_net, decoder_net, _ = load_models()
    test_path = '/mnt/transvae_qm9/test_data/test_batch/'

    compute_cov_mat(test_path + 'smiles.pkl')
