import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from copy import deepcopy
from src.data_process_utils import mol_to_tensor
from src.misc_utils import create_folder
from src.running_stats import RunningStats


def generate_scaffold(smiles, include_chirality=True):
    """return scaffold string of target molecule"""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold\
        .MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def scaffold_split(df_lipo_prop, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    rng = np.random.RandomState(43)
    scaffolds = defaultdict(list)
    for _, row in df_lipo_prop.iterrows():
        scaffold = generate_scaffold(row.smiles)
        scaffolds[scaffold].append(row.CMPD_CHEMBLID)

    scaffold_sets = rng.permutation(list(scaffolds.values()))
    n_total_valid = int(np.floor(frac_valid * df_lipo_prop.shape[0]))
    n_total_test = int(np.floor(frac_test * df_lipo_prop.shape[0]))

    train_index = []
    valid_index = []
    test_index = []

    for scaffold_set in scaffold_sets:
        if len(valid_index) + len(scaffold_set) <= n_total_valid:
            valid_index.extend(scaffold_set)
        elif len(test_index) + len(scaffold_set) <= n_total_test:
            test_index.extend(scaffold_set)
        else:
            train_index.extend(scaffold_set)

    return np.array(train_index), np.array(valid_index), np.array(test_index)


def get_and_save_data_batch(lipo_df, data_idx, dest_data_path):
    rs = RunningStats(1)
    batch = 0
    for mol_id in data_idx:
        row = lipo_df[lipo_df.CMPD_CHEMBLID == mol_id].iloc[0]
        mol = Chem.MolFromSmiles(row.smiles)
        try:
            g, _ = mol_to_tensor(deepcopy(mol))
        except Exception as e:
            print(e)
            continue

        y = np.array([row[_name] for _name in target_name])
        rs.push(y)

        np.savez_compressed(dest_data_path + f'GDR_{batch}', G=g[..., :-4], Y=y)
        batch += 1
        if batch % 100 == 0:
            rs_mean = rs.mean()
            rs_std = rs.standard_deviation()
            with open(dest_data_path + 'stats.pkl', 'wb') as f:
                pickle.dump(np.vstack((rs_mean, rs_std)), f)
    rs_mean = rs.mean()
    rs_std = rs.standard_deviation()
    with open(dest_data_path + 'stats.pkl', 'wb') as f:
        pickle.dump(np.vstack((rs_mean, rs_std)), f)


if __name__ == "__main__":
    lipo_df = pd.read_csv("./lipo_prop.csv")
    train_idx, val_idx, test_idx = scaffold_split(lipo_df)
    target_name = [
        "exp",
    ]
    train_path = '/mnt/transvae_lipo_prop_scaffold/train_data/train_batch/'
    val_path = '/mnt/transvae_lipo_prop_scaffold/test_data/val_batch/'
    test_path = '/mnt/transvae_lipo_prop_scaffold/test_data/test_batch/'

    create_folder(train_path)
    create_folder(val_path)
    create_folder(test_path)
    get_and_save_data_batch(lipo_df, train_idx, train_path)
    get_and_save_data_batch(lipo_df, val_idx, val_path)
    get_and_save_data_batch(lipo_df, test_idx, test_path)
