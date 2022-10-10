import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from copy import deepcopy
from src.graph_utils import draw_mol_with_idx
from src.data_process_utils import mol_to_tensor
from src.misc_utils import create_folder
from src.running_stats import RunningStats


def generate_scaffold(smiles, include_chirality=True):
    """return scaffold string of target molecule"""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold\
        .MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def scaffold_split(df_qm9_prop, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    rng = np.random.RandomState(43)
    scaffolds = defaultdict(list)
    for _, row in df_qm9_prop.iterrows():
        scaffold = generate_scaffold(row.smiles)
        scaffolds[scaffold].append(row.mol_id)

    scaffold_sets = rng.permutation(list(scaffolds.values()))
    n_total_valid = int(np.floor(frac_valid * df_qm9_prop.shape[0]))
    n_total_test = int(np.floor(frac_test * df_qm9_prop.shape[0]))

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


def get_and_save_data_batch(qm9_df, data_idx, dest_data_path):
    suppl = Chem.SDMolSupplier('gdb9.sdf')
    rs_homo = RunningStats()
    rs_lumo = RunningStats()
    rs_gap = RunningStats()
    batch = 0
    for mol_id in data_idx:
        mol_idx = int(mol_id.replace('gdb_', '')) - 1
        mol = suppl[mol_idx]
        try:
            g, _ = mol_to_tensor(deepcopy(mol))
        except Exception as e:
            # draw_mol_with_idx(mol)
            print(e)
            continue
        row = qm9_df[qm9_df.mol_id == mol_id].iloc[0]
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
    qm9_df = pd.read_csv("./qm9_prop.csv")
    train_idx, val_idx, test_idx = scaffold_split(qm9_df)
    train_path = '/mnt/transvae_qm9_prop_scaffold/train_data/train_batch/'
    val_path = '/mnt/transvae_qm9_prop_scaffold/test_data/val_batch/'
    test_path = '/mnt/transvae_qm9_prop_scaffold/test_data/test_batch/'

    create_folder(train_path)
    create_folder(val_path)
    create_folder(test_path)
    get_and_save_data_batch(qm9_df, train_idx, train_path)
    get_and_save_data_batch(qm9_df, val_idx, val_path)
    get_and_save_data_batch(qm9_df, test_idx, test_path)
