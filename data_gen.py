import re
import json
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from rdkit import Chem
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from src.data_process_utils import mol_to_tensor
from src.graph_utils import draw_mol_with_idx
from src.misc_utils import create_folder, pickle_save, pickle_load
from src.running_stats import RunningStats


def get_train_val_test_smiles():
    qm9_df = pd.read_csv("./qm9_prop.csv")
    qm9_df = shuffle(qm9_df)

    create_folder('/mnt/transvae_qm9_prop/train_data/train_batch/')
    create_folder('/mnt/transvae_qm9_prop/test_data/test_batch/')
    create_folder('/mnt/transvae_qm9_prop/test_data/val_batch/')

    # train, val, test split
    smiles_train, smiles_test \
        = train_test_split(qm9_df, test_size=0.1, random_state=43)

    smiles_train, smiles_val \
        = train_test_split(smiles_train, test_size=0.1, random_state=43)

    smiles_train.to_csv('/mnt/transvae_qm9_prop/train_data/train_batch/df_train.csv')
    smiles_val.to_csv('/mnt/transvae_qm9_prop/test_data/val_batch/df_val.csv')
    smiles_test.to_csv('/mnt/transvae_qm9_prop/test_data/test_batch/df_test.csv')


def get_and_save_data_batch(smiles_path, dest_data_path):
    suppl = Chem.SDMolSupplier('gdb9.sdf')
    df_data = pd.read_csv(smiles_path)
    rs_homo = RunningStats()
    rs_lumo = RunningStats()
    rs_gap = RunningStats()
    batch = 0
    for _, row in df_data.iterrows():
        mol_idx = int(row.mol_id.replace('gdb_', '')) - 1
        mol = suppl[mol_idx]
        try:
            g, _ = mol_to_tensor(deepcopy(mol))
        except Exception as e:
            # draw_mol_with_idx(mol)
            print(e)
            continue
        y = np.array([row.homo, row.lumo, row.gap])
        rs_homo.push(row.homo)
        rs_lumo.push(row.lumo)
        rs_gap.push(row.gap)

        np.savez_compressed(dest_data_path + f'GDR_{batch}', G=g, Y=y)
        batch += 1
        if batch % 100 == 0:
            mean_homo = rs_homo.mean()
            stdev_homo = rs_homo.standard_deviation()
            mean_lumo = rs_lumo.mean()
            stdev_lumo = rs_lumo.standard_deviation()
            mean_gap = rs_gap.mean()
            stdev_gap = rs_gap.standard_deviation()
            with open(dest_data_path + 'stats.pkl', 'wb') as f:
                pickle.dump(np.array([mean_homo, stdev_homo,
                                      mean_lumo, stdev_lumo,
                                      mean_gap, stdev_gap]), f)
    mean_homo = rs_homo.mean()
    stdev_homo = rs_homo.standard_deviation()
    mean_lumo = rs_lumo.mean()
    stdev_lumo = rs_lumo.standard_deviation()
    mean_gap = rs_gap.mean()
    stdev_gap = rs_gap.standard_deviation()
    with open(dest_data_path + 'stats.pkl', 'wb') as f:
        pickle.dump(np.array([mean_homo, stdev_homo,
                              mean_lumo, stdev_lumo,
                              mean_gap, stdev_gap]), f)


if __name__ == "__main__":
    get_train_val_test_smiles()
    get_and_save_data_batch('/mnt/transvae_qm9_prop/train_data/train_batch/df_train.csv',
                            '/mnt/transvae_qm9_prop/train_data/train_batch/')
    get_and_save_data_batch('/mnt/transvae_qm9_prop/test_data/val_batch/df_val.csv',
                            '/mnt/transvae_qm9_prop/test_data/val_batch/')
    get_and_save_data_batch('/mnt/transvae_qm9_prop/test_data/test_batch/df_test.csv',
                            '/mnt/transvae_qm9_prop/test_data/test_batch/')
