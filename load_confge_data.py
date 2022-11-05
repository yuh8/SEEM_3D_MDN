import pickle
import numpy as np


def pickle_load(load_path):
    with open(load_path, 'rb') as handle:
        file = pickle.load(handle)
    return file


def pickle_save(file, save_path):
    with open(save_path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('test_data_200.pkl', 'rb') as fp:
    test_data = pickle.load(fp)

smiles_test = []
for _data in test_data:
    smiles_test.append(_data.smiles)

pickle_save(np.unique(smiles_test), '/mnt/transvae_ablation/test_data/test_batch/smiles_confgf.pkl')

breakpoint()
