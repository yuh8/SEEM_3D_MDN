import pickle
import numpy as np
import torch_geometric


with open('qm9_property.pkl', 'rb') as fp:
    qm9_property_mol_data = pickle.load(fp)

packed_data = {}
for _data in qm9_property_mol_data:
    if _data.smiles in packed_data.keys():
        packed_data[_data.smiles]['confs'].append(np.array(_data.pos))
    else:
        packed_data[_data.smiles] = {}
        packed_data[_data.smiles]['rdmol'] = _data.rdmol
        packed_data[_data.smiles]['confs'] = [np.array(_data.pos)]


with open('packed_qm9_property.pkl', 'wb') as fp:
    pickle.dump(packed_data, fp)

breakpoint()
