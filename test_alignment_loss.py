import json
import pickle
import numpy as np
import pandas as pd
from rdkit.Chem.rdMolAlign import GetAlignmentTransform
from src.misc_utils import pickle_load, kabsch_fit


test_path = 'D:/seem_3d_data/test_data/test_batch/smiles.pkl'
drugs_file = "D:/seem_3d_data/data/rdkit_folder/summary_drugs.json"
with open(drugs_file, "r") as f:
    drugs_summ = json.load(f)

smiles = pickle_load(test_path)
for smi in smiles[:200]:
    try:
        mol_path = "D:/seem_3d_data/data/rdkit_folder/" + drugs_summ[smi]['pickle_path']
        with open(mol_path, "rb") as f:
            mol_dict = pickle.load(f)
    except:
        continue

    conf_df = pd.DataFrame(mol_dict['conformers'])
    conf_df.sort_values(by=['boltzmannweight'], ascending=False, inplace=True)

    if conf_df.shape[0] < 1:
        continue
    mol_prob = conf_df.iloc[0].rd_mol
    mol_ref = conf_df.iloc[3].rd_mol
    rmsd, Rot = GetAlignmentTransform(mol_prob, mol_ref)
    conf_prob = np.expand_dims(mol_prob.GetConformer(0).GetPositions(), axis=0)
    conf_ref = np.expand_dims(mol_ref.GetConformer(0).GetPositions(), axis=0)
    mask = np.ones((1, conf_ref.shape[1], 1))
    R, conf_prob = kabsch_fit(conf_prob, conf_ref, mask)

    diff = conf_prob - conf_ref
    rmsd_pred = np.sqrt((diff * diff).sum() / conf_ref.shape[1])
    print("rmsd = {0}, rmsd_pred ={1}".format(rmsd, rmsd_pred))
