import random
import numpy as np
import networkx as nx
from .xyz2mol import xyz2mol
from .CONSTS import (MAX_NUM_ATOMS, FEATURE_DEPTH,
                     CHARGES, ATOM_LIST, ATOM_HYBR_NAMES,
                     BOND_NAMES, BOND_STEREO_NAMES)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=1000)


def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   atom_charge=atom.GetFormalCharge(),
                   atom_hybrd=atom.GetHybridization(),
                   atom_symbol=atom.GetSymbol())

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType(),
                   bond_stereo=bond.GetStereo())

    return G


def verify_mol(mol):
    '''
    https://github.com/jensengroup/xyz2mol
    '''
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    conf = mol.GetConformer(0)
    coordinates = conf.GetPositions()
    coordinates = np.array(coordinates)
    data = {
        'atoms': atoms,
        'coordinates': coordinates,
        'charge': 0,
        'allow_charged_fragments': False,
        'use_graph': True,
        'use_huckel': False,
        'embed_chiral': True,
    }
    try:
        return xyz2mol(**data)
    except:
        return None


def mol_to_tensor(mol, training=True):
    smi_graph = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
    atom_indices = []
    elements = []
    charges = []
    hybrds = []
    for atom in mol.GetAtoms():
        atom_indices.append(atom.GetIdx())
        elements.append(atom.GetSymbol())
        charges.append(atom.GetFormalCharge())
        hybrds.append(atom.GetHybridization())

    bond_channel_start = len(ATOM_LIST) + len(CHARGES) + len(ATOM_HYBR_NAMES)
    coords = np.zeros((MAX_NUM_ATOMS, 4))

    # get conformer
    conf = mol.GetConformer(0)

    # shuffle atom sequence as data augumentation
    if training:
        random.shuffle(atom_indices)
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
            pos_ii = conf.GetAtomPosition(ii)
            pos_jj = conf.GetAtomPosition(jj)
            dist = np.linalg.norm(pos_ii - pos_jj)
            feature_vec[-1] = dist

            smi_graph[idx, idy, :] = feature_vec
            smi_graph[idy, idx, :] = feature_vec

    return smi_graph, coords


def connect_3rd_neighbour(mol, smi_graph):
    '''
    Breadth first 2nd and 3rd neighbor search
    ##########################################
    Direct neighbours: atoms that are directly connected
    2nd neighbours: atoms that are connected to direct neighbours but are not direct neighbours
    3rd neighbours: atoms connected to 2nd neighbours but are neither direct nor 2nd neighbours
    '''
    mol_len = len(mol.GetAtoms())
    breakpoint()
    connect_map = smi_graph[..., :-1].sum(-1)
    np.fill_diagonal(connect_map, 0)
    virtual_bond_channel_start = len(ATOM_LIST) + len(CHARGES) + len(ATOM_HYBR_NAMES) + len(BOND_NAMES)
    third_neighbours = []
    for ii in range(mol_len):
        # direct neighbnours
        connect_atom_idx_1sts = np.where(connect_map[ii, :] > 2)[0]
        num_neighbors = connect_atom_idx_1sts.shape[0]
        connect_atom_idx_2nds = np.array([])
        # 2nd neigbours
        for idx_1st in connect_atom_idx_1sts:
            _connect_atom_idx_2nds = np.where(connect_map[idx_1st, :] > 2)[0]
            _connect_atom_idx_2nds = _connect_atom_idx_2nds[_connect_atom_idx_2nds != ii]
            connect_atom_idx_2nds = np.append(connect_atom_idx_2nds, _connect_atom_idx_2nds)

        # 2nd neighbour should not be direct neighbours
        filter_cond = [idx not in connect_atom_idx_1sts for idx in connect_atom_idx_2nds]
        connect_atom_idx_2nds = connect_atom_idx_2nds[filter_cond]
        if connect_atom_idx_2nds.shape[0] == 0:
            continue

        connect_atom_idx_2nds = np.unique(connect_atom_idx_2nds)
        connect_atom_idx_2nds = connect_atom_idx_2nds.astype(int)
        for idx_2nd in connect_atom_idx_2nds:
            feature_vec = smi_graph[ii, idx_2nd]
            feature_vec[virtual_bond_channel_start] = 1
            smi_graph[ii, idx_2nd] = feature_vec
            smi_graph[idx_2nd, ii] = feature_vec

        if num_neighbors >= 3 or ii in third_neighbours:
            continue

        # randomly choose a single 3rd neighbour
        random.shuffle(connect_atom_idx_2nds)
        for idx_2nd in connect_atom_idx_2nds:
            connect_atom_idx_3rds = np.where(connect_map[idx_2nd, :] > 2)[0]
            # 3rd neighbours should be neither 2nd or direct neighbours
            filter_idx = [idx not in connect_atom_idx_1sts and idx not in connect_atom_idx_2nds for idx in connect_atom_idx_3rds]
            connect_atom_idx_3rds = connect_atom_idx_3rds[filter_idx]
            if connect_atom_idx_3rds.shape[0] == 0:
                continue
            random.shuffle(connect_atom_idx_3rds)
            connect_atom_idx_3rd = connect_atom_idx_3rds[0]
            feature_vec = smi_graph[ii, connect_atom_idx_3rd]
            feature_vec[virtual_bond_channel_start + 1] = 1
            smi_graph[ii, connect_atom_idx_3rd] = feature_vec
            smi_graph[connect_atom_idx_3rd, ii] = feature_vec
            third_neighbours.append(connect_atom_idx_3rd)
            break
    d = smi_graph[..., -1]
    mask = smi_graph[..., :-1].sum(-1) > 2
    d *= mask
    smi_graph[..., -1] = d
    return smi_graph
