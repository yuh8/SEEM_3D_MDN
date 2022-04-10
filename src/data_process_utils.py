import random
import numpy as np
from src.CONSTS import (MAX_NUM_ATOMS, FEATURE_DEPTH,
                        CHARGES, ATOM_LIST, ATOM_HYBR_NAMES,
                        BOND_NAMES, BOND_STEREO_NAMES)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=1000)


def mol_to_tensor(mol, training=True):
    smi_graph = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
    atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    hybrds = [atom.GetHybridization()for atom in mol.GetAtoms()]
    bond_channel_start = len(ATOM_LIST) + len(CHARGES) + len(ATOM_HYBR_NAMES)
    coords = np.zeros((MAX_NUM_ATOMS, 4))

    # get positions
    conf = mol.GetConformers()[0]
    pos = conf.GetPositions()
    pos = np.hstack((np.array(atom_indices).reshape(-1, 1) + 1, pos))
    coords[:pos.shape[0], ...] = pos

    # shuffle atom sequence as data augumentation
    if training:
        random.shuffle(atom_indices)
    pos = pos[atom_indices, ...]
    for idx, ii in enumerate(atom_indices):
        for idy, jj in enumerate(atom_indices):
            feature_vec = np.zeros(FEATURE_DEPTH)
            if idx == idy:
                # charge
                charge_idx = len(ATOM_LIST) + CHARGES.index(charges[ii])
                feature_vec[charge_idx] = 1
                # hybridization
                hybrd_idx = len(ATOM_LIST) + len(CHARGES) + ATOM_HYBR_NAMES.index(hybrds[ii])
                feature_vec[hybrd_idx] = 1

            if idx > idy:
                continue

            atom_idx_ii = ATOM_LIST.index(elements[ii])
            atom_idx_jj = ATOM_LIST.index(elements[jj])
            feature_vec[atom_idx_ii] += 1
            feature_vec[atom_idx_jj] += 1
            if mol.GetBondBetweenAtoms(ii, jj) is not None:
                # bond type
                bond_name = mol.GetBondBetweenAtoms(ii, jj).GetBondType()
                bond_idx = BOND_NAMES.index(bond_name)
                bond_feature_idx = bond_channel_start + bond_idx
                feature_vec[bond_feature_idx] = 1
                # stereo type
                stereo_name = mol.GetBondBetweenAtoms(ii, jj).GetStereo()
                stereo_idx = BOND_STEREO_NAMES.index(stereo_name)
                stereo_feature_idx = bond_channel_start + len(BOND_NAMES) + stereo_idx
                feature_vec[stereo_feature_idx] = 1
            # distance
            dist = np.linalg.norm(pos[idx, 1:] - pos[idy, 1:])
            feature_vec[-1] = dist

            smi_graph[idx, idy, :] = feature_vec
            feature_vec[-1] = 0
            smi_graph[idy, idx, :] = feature_vec

    return smi_graph, coords


def connect_3rd_neighbour(mol, smi_graph):
    '''
    Breadth first 3rd neighbor search
    '''
    mol_len = len([atom.GetIdx() for atom in mol.GetAtoms()])
    connect_map = smi_graph[..., :-1].sum(-1) - 2
    np.fill_diagonal(connect_map, 0)
    virtual_bond_channel_start = len(ATOM_LIST) + len(CHARGES) + len(ATOM_HYBR_NAMES) + len(BOND_NAMES)
    third_neighbours = []
    for ii in range(mol_len):
        feature_vec = np.zeros(FEATURE_DEPTH)
        connect_atom_idx_1sts = np.where(connect_map[ii, :] > 0)[0]
        num_neighbors = connect_atom_idx_1sts.shape[0]
        connect_atom_idx_2nds = np.array([])
        for idx_1st in connect_atom_idx_1sts:
            _connect_atom_idx_2nds = np.where(connect_map[idx_1st, :] > 0)[0]
            _connect_atom_idx_2nds = _connect_atom_idx_2nds[_connect_atom_idx_2nds != ii]
            connect_atom_idx_2nds = np.append(connect_atom_idx_2nds, _connect_atom_idx_2nds)
        if connect_atom_idx_2nds.shape[0] == 0:
            continue

        connect_atom_idx_2nds = np.unique(connect_atom_idx_2nds)
        connect_atom_idx_2nds = connect_atom_idx_2nds.astype(int)
        feature_vec[virtual_bond_channel_start] = 1
        for idx_2nd in connect_atom_idx_2nds:
            smi_graph[ii, idx_2nd] = feature_vec
            smi_graph[idx_2nd, ii] = feature_vec

        if num_neighbors >= 3 or ii in third_neighbours:
            continue

        feature_vec = np.zeros(FEATURE_DEPTH)
        # only choose a single 3rd neighbour
        for idx_2nd in connect_atom_idx_2nds:
            connect_atom_idx_3rds = np.where(connect_map[idx_2nd, :] > 0)[0]
            filter_idx = [idx not in connect_atom_idx_1sts and idx not in connect_atom_idx_2nds for idx in connect_atom_idx_3rds]
            connect_atom_idx_3rds = connect_atom_idx_3rds[filter_idx]
            if connect_atom_idx_3rds.shape[0] == 0:
                continue
            connect_atom_idx_3rd = connect_atom_idx_3rds[0]
            feature_vec[virtual_bond_channel_start + 1] = 1
            smi_graph[ii, connect_atom_idx_3rd] = feature_vec
            smi_graph[connect_atom_idx_3rd, ii] = feature_vec
            third_neighbours.append(connect_atom_idx_3rd)
            break
    return smi_graph
