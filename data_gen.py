from copy import deepcopy
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_process_utils import mol_to_tensor
from src.graph_utils import draw_mol_with_idx
from src.misc_utils import create_folder, pickle_save, pickle_load
from src.CONSTS import NUM_CONFS_PER_MOL


def get_train_val_test_smiles():
    drugs_file = "/mnt/raw_data/rdkit_folder/summary_qm9.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    all_simles = list(drugs_summ.keys())
    np.random.shuffle(all_simles)
    create_folder('/mnt/raw_data/transvae_qm9/train_data/train_batch/')
    create_folder('/mnt/raw_data/transvae_qm9/test_data/test_batch/')
    create_folder('/mnt/raw_data/transvae_qm9/test_data/val_batch/')

    # train, val, test split
    smiles_train, smiles_test \
        = train_test_split(all_simles, test_size=0.1, random_state=43)

    smiles_train, smiles_val \
        = train_test_split(smiles_train, test_size=0.1, random_state=43)

    pickle_save(smiles_train, '/mnt/raw_data/transvae_qm9/train_data/train_batch/smiles.pkl')
    pickle_save(smiles_test, '/mnt/raw_data/transvae_qm9/test_data/test_batch/smiles.pkl')
    pickle_save(smiles_val, '/mnt/raw_data/transvae_qm9/test_data/val_batch/smiles.pkl')


def get_and_save_data_batch(smiles_path, dest_data_path, batch_num=200000):
    drugs_file = "/mnt/raw_data/rdkit_folder/summary_qm9.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    smiles = pickle_load(smiles_path)
    batch = 0
    for smi in smiles:
        try:
            mol_path = "/mnt/raw_data/rdkit_folder/" + drugs_summ[smi]['pickle_path']
            with open(mol_path, "rb") as f:
                mol_dict = pickle.load(f)
        except Exception as e:
            print(e)
            continue

        conf_df = pd.DataFrame(mol_dict['conformers'])
        conf_df.sort_values(by=['boltzmannweight'], ascending=False, inplace=True)
        if conf_df.shape[0] < 1:
            continue
        for _, mol_row in conf_df.iloc[:NUM_CONFS_PER_MOL, :].iterrows():
            mol = mol_row.rd_mol

            try:
                g, r = mol_to_tensor(mol)
            except Exception as e:
                # draw_mol_with_idx(mol)
                print(e)
                continue

            np.savez_compressed(dest_data_path + f'GDR_{batch}', G=g, R=r)
            batch += 1
            if batch == batch_num:
                break
        else:
            continue
        break


def get_num_of_disconnected_graphs():
    drugs_file = "/mnt/raw_data/rdkit_folder/summary_qm9.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    all_simles = list(drugs_summ.keys())
    cnt_disconnected_graphs = 0
    for idx, smi in enumerate(all_simles):
        try:
            mol_path = "/mnt/raw_data/rdkit_folder/" + drugs_summ[smi]['pickle_path']
            with open(mol_path, "rb") as f:
                mol_dict = pickle.load(f)
        except Exception as e:
            print(e)
            continue

        conf_df = pd.DataFrame(mol_dict['conformers'])
        conf_df.sort_values(by=['boltzmannweight'], ascending=False, inplace=True)
        mol_row = conf_df.iloc[0]
        mol = mol_row.rd_mol

        try:
            g, r = mol_to_tensor(mol)
        except Exception as e:
            # draw_mol_with_idx(mol)
            cnt_disconnected_graphs += 1

        if idx % 1000 == 0:
            print(f'percentage of disconnected graphs = {np.round(cnt_disconnected_graphs/(idx+1),4)}')


if __name__ == "__main__":
    # get_num_of_disconnected_graphs()
    get_train_val_test_smiles()
    # get_and_save_data_batch('/mnt/transvae_qm9/train_data/train_batch/smiles.pkl',
    #                         '/mnt/transvae_qm9/train_data/train_batch/')
    # get_and_save_data_batch('/mnt/transvae_qm9/test_data/val_batch/smiles.pkl',
    #                         '/mnt/transvae_qm9/test_data/val_batch/', batch_num=2500)
    get_and_save_data_batch('/mnt/raw_data/transvae_qm9/test_data/test_batch/smiles.pkl',
                            '/mnt/raw_data/transvae_qm9/test_data/test_batch/', batch_num=20000)
