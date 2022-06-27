import json
import pickle
import numpy as np
import pandas as pd
import sparse as sp
from sklearn.model_selection import train_test_split
from src.data_process_utils import mol_to_tensor
from src.graph_utils import draw_mol_with_idx
from src.misc_utils import create_folder, pickle_save, pickle_load, RunningStats
from src.CONSTS import BATCH_SIZE, NUM_CONFS_PER_MOL


def get_train_val_test_smiles():
    drugs_file = "D:/seem_3d_data/data/rdkit_folder/summary_drugs.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    all_simles = list(drugs_summ.keys())
    create_folder('D:/seem_3d_data/train_data/train_batch/')
    create_folder('D:/seem_3d_data/test_data/test_batch/')
    create_folder('D:/seem_3d_data/test_data/val_batch/')

    # train, val, test split
    smiles_train, smiles_test \
        = train_test_split(all_simles, test_size=0.1, random_state=43)

    smiles_train, smiles_val \
        = train_test_split(smiles_train, test_size=0.1, random_state=43)

    pickle_save(smiles_train, 'D:/seem_3d_data/train_data/train_batch/smiles.pkl')
    pickle_save(smiles_test, 'D:/seem_3d_data/test_data/test_batch/smiles.pkl')
    pickle_save(smiles_val, 'D:/seem_3d_data/test_data/val_batch/smiles.pkl')


def get_and_save_data_batch(smiles_path, dest_data_path, batch_num=6250):
    rs = RunningStats()
    drugs_file = "D:/seem_3d_data/data/rdkit_folder/summary_drugs.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    smiles = pickle_load(smiles_path)
    G = []
    R = []
    batch = 0
    for sim in smiles:
        try:
            mol_path = "D:/seem_3d_data/data/rdkit_folder/" + drugs_summ[sim]['pickle_path']
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
                draw_mol_with_idx(mol)
                print(e)
                continue

            G.append(g)
            R.append(r)
            if len(G) > BATCH_SIZE:
                _X = sp.COO(np.stack(G[:BATCH_SIZE]))
                _z = sp.COO(np.stack(R[:BATCH_SIZE]))
                _data = (_X, _z)
                with open(dest_data_path + 'GDR_{}.pkl'.format(batch), 'wb') as f:
                    pickle.dump(_data, f)

                G = G[BATCH_SIZE:]
                R = R[BATCH_SIZE:]
                batch += 1
                if batch >= batch_num:
                    break
            if batch >= batch_num:
                break
        if batch >= batch_num:
            break
    if G:
        _X = sp.COO(np.stack(G))
        _z = sp.COO(np.stack(R))
        _data = (_X, _z)
        with open(dest_data_path + 'GDR_{}.pkl'.format(batch), 'wb') as f:
            pickle.dump(_data, f)


def get_num_atoms_dist():
    drugs_file = "D:/seem_3d_data/data/rdkit_folder/summary_drugs.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    all_simles = list(drugs_summ.keys())
    num_atoms = []
    for idx, sim in enumerate(all_simles):
        try:
            mol_path = "D:/seem_3d_data/data/rdkit_folder/" + drugs_summ[sim]['pickle_path']
            with open(mol_path, "rb") as f:
                mol_dict = pickle.load(f)
        except Exception as e:
            print(e)
            continue

        conf_df = pd.DataFrame(mol_dict['conformers'])
        conf_df.sort_values(by=['boltzmannweight'], ascending=False, inplace=True)
        if conf_df.shape[0] < 1:
            continue
        num_atoms.append(conf_df.iloc[0].rd_mol.GetNumAtoms())

        if len(num_atoms) % 1000 == 0:
            pct_95 = np.percentile(num_atoms, 95)
            print("{0}/{1} done with 95 pct {2}".format(idx, len(all_simles), pct_95))

    breakpoint()


if __name__ == "__main__":
    # get_num_atoms_dist()
    get_train_val_test_smiles()
    get_and_save_data_batch('D:/seem_3d_data/train_data/train_batch/smiles.pkl',
                            'D:/seem_3d_data/train_data/train_batch/')
    get_and_save_data_batch('D:/seem_3d_data/test_data/val_batch/smiles.pkl',
                            'D:/seem_3d_data/test_data/val_batch/', batch_num=1000)
    get_and_save_data_batch('D:/seem_3d_data/test_data/test_batch/smiles.pkl',
                            'D:/seem_3d_data/test_data/test_batch/', batch_num=2000)
