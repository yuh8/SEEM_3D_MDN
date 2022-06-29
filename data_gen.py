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
    drugs_file = "/mnt/rdkit_folder/summary_drugs.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    all_simles = list(drugs_summ.keys())
    create_folder('/mnt/seem_3d_data/train_data/train_batch/')
    create_folder('/mnt/seem_3d_data/test_data/test_batch/')
    create_folder('/mnt/seem_3d_data/test_data/val_batch/')

    # train, val, test split
    smiles_train, smiles_test \
        = train_test_split(all_simles, test_size=0.1, random_state=43)

    smiles_train, smiles_val \
        = train_test_split(smiles_train, test_size=0.1, random_state=43)

    pickle_save(smiles_train, '/mnt/seem_3d_data/train_data/train_batch/smiles.pkl')
    pickle_save(smiles_test, '/mnt/seem_3d_data/test_data/test_batch/smiles.pkl')
    pickle_save(smiles_val, '/mnt/seem_3d_data/test_data/val_batch/smiles.pkl')


def get_and_save_data_batch(smiles_path, dest_data_path, batch_num=10000):
    rs = RunningStats()
    drugs_file = "/mnt/rdkit_folder/summary_drugs.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    smiles = pickle_load(smiles_path)
    G = []
    D = []
    R = []
    batch = 0
    for sim in smiles:
        try:
            mol_path = "/mnt/rdkit_folder/" + drugs_summ[sim]['pickle_path']
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
                g, d, _ = mol_to_tensor(mol)
                # cacluate mean and std online for distance
                con_map = g.sum(-1)
                con_d = d[con_map > 3]
                for _d in con_d:
                    rs.push(_d)
            except Exception as e:
                draw_mol_with_idx(mol)
                print(e)
                continue

            G.append(g)
            D.append(d)
            if len(G) > BATCH_SIZE:
                _X = sp.COO(np.stack(G[:BATCH_SIZE]))
                _y = sp.COO(np.stack(D[:BATCH_SIZE]))
                _data = (_X, _y)
                with open(dest_data_path + 'GDR_{}.pkl'.format(batch), 'wb') as f:
                    pickle.dump(_data, f)

                mean = rs.mean()
                stdev = rs.standard_deviation()
                with open(dest_data_path + 'stats.pkl', 'wb') as f:
                    pickle.dump(np.array([mean, stdev]), f)
                G = G[BATCH_SIZE:]
                D = D[BATCH_SIZE:]
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
        _y = sp.COO(np.stack(D))
        _data = (_X, _y)
        with open(dest_data_path + 'GDR_{}.pkl'.format(batch), 'wb') as f:
            pickle.dump(_data, f)

        mean = rs.mean()
        stdev = rs.standard_deviation()
        with open(dest_data_path + 'stats.pkl', 'wb') as f:
            pickle.dump(np.array([mean, stdev]), f)


if __name__ == "__main__":
    get_train_val_test_smiles()
    get_and_save_data_batch('/mnt/seem_3d_data/train_data/train_batch/smiles.pkl',
                            '/mnt/seem_3d_data/train_data/train_batch/')
    get_and_save_data_batch('/mnt/seem_3d_data/test_data/val_batch/smiles.pkl',
                            '/mnt/seem_3d_data/test_data/val_batch/', batch_num=5000)
    get_and_save_data_batch('/mnt/seem_3d_data/test_data/test_batch/smiles.pkl',
                            '/mnt/seem_3d_data/test_data/test_batch/', batch_num=10000)
