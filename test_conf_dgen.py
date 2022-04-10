import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from multiprocessing import freeze_support
from rdkit.Chem.Draw import MolsToGridImage
from src.data_process_utils import mol_to_tensor
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


def get_test_mol(smiles_path):
    mols = []
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
        for _, mol_row in conf_df.iloc[:NUM_CONFS_PER_MOL, :].iterrows():
            mol = mol_row.rd_mol
            G, _ = mol_to_tensor(mol, training=False)
            G_in = G[..., :-1]
            G_in = np.expand_dims(G_in, axis=0)
            # d = np.expand_dims(d, axis=-1)
            d_pred = model(G_in, training=False).numpy()[0]
            d_pred_mean = d_pred[..., 1]
            d_pred_std = np.exp(d_pred[..., 2])

            embed_conformer(mol, d_pred_mean, d_pred_std, seed=43)
            align_conformers(mol)
            plot = MolsToGridImage([mol])
            plot.show()
            breakpoint()
            mols.append(mol)
            if len(mols) % 10 == 0:
                pickle_save(mols, 'generated_confs.pkl')


if __name__ == "__main__":
    freeze_support()
    create_folder('gen_samples/')
    model = load_json_model("conf_model_d_2022-04-09/conf_model_d.json")
    model.compile(optimizer='adam',
                  loss=loss_func)
    model.load_weights("./checkpoints/generator_d_2022-04-09/")
    breakpoint()

    get_test_mol('D:/seem_3d_data/test_data/test_batch/smiles.pkl')
