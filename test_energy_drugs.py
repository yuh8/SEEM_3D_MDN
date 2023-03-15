import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import json
from random import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy
from psikit import Psikit
from tqdm.auto import tqdm
from multiprocessing import freeze_support
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from src.data_process_utils import mol_to_tensor
from src.misc_utils import pickle_load
from src.CONSTS import HIDDEN_SIZE, MAX_NUM_ATOMS, TF_EPS


def load_models():
    g_net = tf.keras.models.load_model('g_net_qm9/GNet/')
    gr_net = tf.keras.models.load_model('gr_net_qm9/GDRNet/')
    decoder_net = tf.keras.models.load_model('dec_net_qm9/DecNet/')
    return g_net, decoder_net, gr_net


def get_prediction(mol, sample_size, g_net, decoder_net):
    mol_origin = deepcopy(mol)
    gr, _ = mol_to_tensor(mol_origin)
    g = np.expand_dims(gr, axis=0)[..., :-4]
    mask = np.sum(np.abs(g), axis=-1)
    mask = np.sum(mask, axis=1, keepdims=True) <= 0
    mask = np.expand_dims(mask, axis=1).astype(np.float32)
    with tf.device('/gpu:1'):
        h = g_net.predict(g)
    h = np.tile(h, [sample_size, 1, 1])
    mask = np.tile(mask, [sample_size, 1, 1, 1])
    z = np.random.normal(0, 1, size=(sample_size, MAX_NUM_ATOMS, HIDDEN_SIZE))
    with tf.device('/gpu:1'):
        r_pred = decoder_net.predict([h, mask, z]) * (1 - np.squeeze(mask)[..., np.newaxis])
    return r_pred


def get_mol_probs(mol_pred, r_pred, num_gens, FF=True):
    mol_probs = []
    for j in tqdm(range(num_gens)):
        mol_prob = deepcopy(mol_pred)
        _conf = mol_prob.GetConformer(0)
        for i in range(mol_prob.GetNumAtoms()):
            _conf.SetAtomPosition(i, r_pred[j][i].tolist())
        if FF:
            MMFFOptimizeMolecule(mol_prob)
        mol_probs.append(mol_prob)
    return mol_probs


class PropertyCalculator(object):
    def __init__(self, threads, memory, seed):
        super().__init__()
        self.pk = Psikit(threads=threads, memory=memory)
        self.seed = seed

    def __call__(self, rdmols):
        mol_ensemble_energy = []
        mol_ensemble_homo = []
        mol_ensemble_lumo = []
        mol_ensemble_dipo = []
        for rdmol in tqdm(rdmols):
            self.pk.mol = rdmol
            try:
                energy, homo, lumo, dipo = self.pk.energy(), self.pk.HOMO, self.pk.LUMO, self.pk.dipolemoment[-1]
                mol_ensemble_energy.append(energy)
                mol_ensemble_homo.append(homo)
                mol_ensemble_lumo.append(lumo)
                mol_ensemble_dipo.append(dipo)
            except:
                pass
        out_data = {}
        out_data['energy'] = np.array(mol_ensemble_energy)
        out_data['homo'] = np.array(mol_ensemble_homo)
        out_data['lumo'] = np.array(mol_ensemble_lumo)
        out_data['dipo'] = np.array(mol_ensemble_dipo)
        return out_data


def get_ensemble_energy(out_data):
    """
    Args:
        props: (4, num_confs)
    """
    avg_ener = np.mean(out_data['energy'])
    low_ener = np.min(out_data['energy'])
    gaps = np.abs(out_data['homo'] - out_data['lumo'])
    avg_gap = np.mean(gaps)
    min_gap = np.min(gaps)
    max_gap = np.max(gaps)
    return np.array([
        avg_ener, low_ener, avg_gap, min_gap, max_gap,
    ])


def compute_energy_stats(smiles_path, g_net, decoder_net):
    drugs_file = "/mnt/raw_data/rdkit_folder/summary_drugs.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    prop_cal = PropertyCalculator(threads=16, memory=24, seed=2021)

    smiles = pickle_load(smiles_path)
    all_diff = []
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

        if num_refs > 100:
            continue

        num_gens = num_refs * 2

        mol_pred = deepcopy(conf_df.iloc[0].rd_mol)

        try:
            r_pred = get_prediction(mol_pred, num_gens, g_net, decoder_net)
        except:
            continue

        mol_probs = get_mol_probs(mol_pred, r_pred, num_gens, FF=False)
        mol_refs = [mol_row for _, mol_row in conf_df.iterrows()]

        prop_gen = get_ensemble_energy(prop_cal(mol_probs)) * 27.211
        prop_gts = get_ensemble_energy(prop_cal(mol_refs)) * 27.211
        prop_diff = np.abs(prop_gts - prop_gen)

        print('\nProperty: %s' % smi)
        print('  Gts :', prop_gts)
        print('  Gen :', prop_gen)
        print('  Diff:', prop_diff)
        all_diff.append(prop_diff.reshape(1, -1))
    all_diff = np.vstack(all_diff)  # (num_mols, 4)
    print(all_diff.shape)

    print('[Difference]')
    print('  Mean:  ', np.mean(all_diff, axis=0))
    print('  Median:', np.median(all_diff, axis=0))
    print('  Std:   ', np.std(all_diff, axis=0))


if __name__ == "__main__":
    freeze_support()
    g_net, decoder_net, _ = load_models()
