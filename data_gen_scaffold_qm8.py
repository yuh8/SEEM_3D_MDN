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


def scaffold_split(df_qm8_prop, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    rng = np.random.RandomState(43)
    suppl = Chem.SDMolSupplier('qm8.sdf')
    scaffolds = defaultdict(list)
    for _, row in df_qm8_prop.iterrows():
        try:
            smi = Chem.MolToSmiles(suppl[int(row.gdb9_index - 1)])
        except:
            continue
        scaffold = generate_scaffold(smi)
        scaffolds[scaffold].append(row.gdb9_index)

    scaffold_sets = rng.permutation(list(scaffolds.values()))
    n_total_valid = int(np.floor(frac_valid * df_qm8_prop.shape[0]))
    n_total_test = int(np.floor(frac_test * df_qm8_prop.shape[0]))

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


def get_and_save_data_batch(qm8_df, data_idx, dest_data_path):
    suppl = Chem.SDMolSupplier('qm8.sdf')
    rs = RunningStats(len(target_name))
    batch = 0
    for mol_id in data_idx:
        mol_idx = int(mol_id) - 1
        mol = suppl[mol_idx]
        try:
            g, _ = mol_to_tensor(deepcopy(mol))
        except Exception as e:
            print(e)
            continue
        row = qm8_df[qm8_df.gdb9_index == mol_id].iloc[0]
        y = np.array([row[_name] for _name in target_name])
        rs.push(y)

        np.savez_compressed(dest_data_path + f'GDR_{batch}', G=g, Y=y)
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
    qm8_df = pd.read_csv("./qm8_prop.csv")
    train_idx, val_idx, test_idx = scaffold_split(qm8_df)
    breakpoint()

    target_name = [
        "E1-CC2",
        "E2-CC2",
        "f1-CC2",
        "f2-CC2",
        "E1-PBE0",
        "E2-PBE0",
        "f1-PBE0",
        "f2-PBE0",
        "E1-CAM",
        "E2-CAM",
        "f1-CAM",
        "f2-CAM",
    ]
    train_path = '/mnt/transvae_qm8_prop_scaffold/train_data/train_batch/'
    val_path = '/mnt/transvae_qm8_prop_scaffold/test_data/val_batch/'
    test_path = '/mnt/transvae_qm8_prop_scaffold/test_data/test_batch/'

    create_folder(train_path)
    create_folder(val_path)
    create_folder(test_path)
    get_and_save_data_batch(qm8_df, train_idx, train_path)
    get_and_save_data_batch(qm8_df, val_idx, val_path)
    get_and_save_data_batch(qm8_df, test_idx, test_path)
