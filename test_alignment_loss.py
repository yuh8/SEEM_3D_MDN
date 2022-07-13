import json
import pickle
import numpy as np
import pandas as pd
from rdkit.Chem.rdMolAlign import GetAlignmentTransform
from src.misc_utils import pickle_load, kabsch_fit, kabsch_rmsd


test_path = '/mnt/transvae/test_data/test_batch/smiles.pkl'
drugs_file = "/mnt/rdkit_folder/summary_drugs.json"
with open(drugs_file, "r") as f:
    drugs_summ = json.load(f)

smiles = pickle_load(test_path)
for smi in smiles[:200]:
    try:
        mol_path = "/mnt/rdkit_folder/" + drugs_summ[smi]['pickle_path']
        with open(mol_path, "rb") as f:
            mol_dict = pickle.load(f)
    except:
        continue

    conf_df = pd.DataFrame(mol_dict['conformers'])
    conf_df.sort_values(by=['boltzmannweight'], ascending=False, inplace=True)

    if conf_df.shape[0] < 5:
        continue
    mol_prob = [conf_df.iloc[0].rd_mol,
                conf_df.iloc[1].rd_mol,
                conf_df.iloc[2].rd_mol]
    mol_ref = [conf_df.iloc[3].rd_mol,
               conf_df.iloc[4].rd_mol,
               conf_df.iloc[5].rd_mol]
    rmsd = np.zeros(3)
    conf_prob = np.zeros((3, 120, 3))
    conf_ref = np.zeros((3, 120, 3))
    mask = np.ones((3, 120, 1))
    for i in range(3):
        rmsd[i], Rot = GetAlignmentTransform(mol_prob[i], mol_ref[i])
        num_atoms = mol_prob[i].GetNumAtoms()
        conf_prob[i][:num_atoms, ...] = mol_prob[i].GetConformer(0).GetPositions()
        conf_ref[i][:num_atoms, ...] = mol_ref[i].GetConformer(0).GetPositions()
        mask[i][num_atoms:] = 0

    rmsd_pred = kabsch_rmsd(conf_prob, conf_ref, mask)
    print("rmsd = {0}, rmsd_pred ={1}".format(rmsd, rmsd_pred))
    breakpoint()
