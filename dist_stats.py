import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_process_utils import mol_to_d_dist
from src.misc_utils import pickle_load


smiles_path = 'D:/seem_3d_data/train_data/train_batch/smiles.pkl'
drugs_file = "D:/seem_3d_data/data/rdkit_folder/summary_drugs.json"
with open(drugs_file, "r") as f:
    drugs_summ = json.load(f)

smiles = pickle_load(smiles_path)
G = []
D = []
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
    if conf_df.shape[0] < 100:
        continue
    dist_stats = mol_to_d_dist(conf_df)
    for i in range(200):
        if len(dist_stats[i]) >= 10:
            _df = pd.DataFrame(dist_stats[i], columns=["dist_{}".format(i)])
            sns.displot(_df, x="dist_{}".format(i), kde=True)
            plt.show()
