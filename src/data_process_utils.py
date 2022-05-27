import random
import numpy as np
import networkx as nx
from .xyz2mol import xyz2mol
from .CONSTS import (BOND_RINGTYPE_SIZE,
                     MAX_NUM_ATOMS, FEATURE_DEPTH,
                     CHARGES, ATOM_LIST, ATOM_CHIR_NAMES,
                     BOND_NAMES, BOND_STEREO_NAMES, RING_SIZES,
                     ATOM_TYPE_SIZE, CHARGE_TYPE_SIZE, CHIR_TYPE_SIZE,
                     BOND_TYPE_SIZE, BOND_STEREOTYPE_SIZE)
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


def get_atom_channel_feature(element, charge, chir):
    # element
    atom_idx = ATOM_LIST.index(element)
    atom_type_channel = np.zeros(ATOM_TYPE_SIZE)
    atom_type_channel[atom_idx] = 1

    # charge
    charge_idx = CHARGES.index(charge)
    charge_type_channel = np.zeros(CHARGE_TYPE_SIZE)
    charge_type_channel[charge_idx] = 1

    # chiral
    chir_idx = ATOM_CHIR_NAMES.index(chir)
    chir_type_channel = np.zeros(CHIR_TYPE_SIZE)
    chir_type_channel[chir_idx] = 1

    return np.hstack([atom_type_channel,
                      charge_type_channel,
                      chir_type_channel])


def get_bond_channel_feature(bond_idx=None,
                             stereo_idx=None,
                             ring_indices=[]):

    bond_type_channel = np.zeros(BOND_TYPE_SIZE)
    bond_stereo_channel = np.zeros(BOND_STEREOTYPE_SIZE)
    bond_ring_channel = np.zeros(BOND_RINGTYPE_SIZE)
    if bond_idx is not None:
        bond_type_channel[bond_idx] = 1
        bond_stereo_channel[stereo_idx] = 1

        for ring_idx in ring_indices:
            bond_ring_channel[ring_idx] = 1

    return np.hstack([bond_type_channel,
                      bond_stereo_channel,
                      bond_ring_channel])


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
        mol = xyz2mol(**data)
    except Exception as e:
        print(e)
        return False

    return True


def mol_to_tensor(mol, training=True):
    smi_graph = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH))
    d = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS))
    atom_indices = []
    elements = []
    charges = []
    chirs = []
    for atom in mol.GetAtoms():
        atom_indices.append(atom.GetIdx())
        elements.append(atom.GetSymbol())
        charges.append(atom.GetFormalCharge())
        chirs.append(atom.GetChiralTag())

    # get conformer
    conf = mol.GetConformer(0)

    # shuffle atom sequence as data augumentation
    if training:
        random.shuffle(atom_indices)
    for idx, ii in enumerate(atom_indices):
        for idy, jj in enumerate(atom_indices):
            if idx == idy:
                atom_channel_feature = get_atom_channel_feature(elements[ii], charges[ii], chirs[ii])
                atom_channel_len = len(atom_channel_feature)
                smi_graph[idx, idy, :atom_channel_len] = atom_channel_feature

            if idx > idy:
                continue

            if mol.GetBondBetweenAtoms(ii, jj) is not None:
                bond = mol.GetBondBetweenAtoms(ii, jj)
                # atom channel feature
                atom_channel_feature_ii = get_atom_channel_feature(elements[ii],
                                                                   charges[ii], chirs[ii])
                atom_channel_feature_jj = get_atom_channel_feature(elements[jj],
                                                                   charges[jj], chirs[jj])
                atom_channel_feature = atom_channel_feature_ii + atom_channel_feature_jj

                # bond type
                bond_name = bond.GetBondType()
                bond_idx = BOND_NAMES.index(bond_name)

                # stereo type
                stereo_name = bond.GetStereo()
                stereo_idx = BOND_STEREO_NAMES.index(stereo_name)

                # ring size
                ring_indices = [idx + 1 for idx, size in enumerate(RING_SIZES) if bond.IsInRingSize(size)]
                if len(ring_indices) == 0:
                    ring_indices = [0]

                bond_channel_feature = get_bond_channel_feature(bond_idx, stereo_idx, ring_indices)

                # update graph
                feature_vec = np.hstack((atom_channel_feature, bond_channel_feature))
                smi_graph[idx, idy, :] = feature_vec
                smi_graph[idy, idx, :] = feature_vec
                assert feature_vec.sum() >= 9, 'Wrong bond feature creation'
            # distance
            pos_ii = conf.GetAtomPosition(ii)
            pos_jj = conf.GetAtomPosition(jj)
            dist = np.linalg.norm(pos_ii - pos_jj)
            d[idx, idy] = dist
            d[idy, idx] = dist
    return smi_graph, d


def update_virtual_bond_feature(smi_graph, row, col, bond_idx):
    _feature_vec = smi_graph[row, col, :]
    if _feature_vec.sum() >= 9:  # if a bond already exists
        return smi_graph

    bond_start = ATOM_TYPE_SIZE + CHARGE_TYPE_SIZE + CHIR_TYPE_SIZE
    bond_channel_feature = get_bond_channel_feature(bond_idx=bond_idx, stereo_idx=0,
                                                    ring_indices=[0])
    atom_channel_feature_ii = smi_graph[row, row, :bond_start]
    atom_channel_feature_jj = smi_graph[col, col, :bond_start]
    atom_channel_feature = atom_channel_feature_ii + atom_channel_feature_jj

    feature_vec = np.hstack((atom_channel_feature, bond_channel_feature))
    smi_graph[row, col, :] = feature_vec
    smi_graph[col, row, :] = feature_vec
    return smi_graph


def connect_3rd_neighbour(mol, smi_graph):
    '''
    Breadth first 2nd and 3rd neighbor search
    ##########################################
    Direct neighbours: atoms that are directly connected
    2nd neighbours: atoms that are connected to direct neighbours but are not direct neighbours
    3rd neighbours: atoms connected to 2nd neighbours but are neither direct nor 2nd neighbours
    '''
    mol_len = len(mol.GetAtoms())
    connect_map = smi_graph.sum(-1)
    np.fill_diagonal(connect_map, 0)
    third_neighbours = []
    for ii in range(mol_len):
        # direct neighbnours
        connect_atom_idx_1sts = np.where(connect_map[ii, :] > 3)[0]
        num_neighbors = connect_atom_idx_1sts.shape[0]
        connect_atom_idx_2nds = np.array([])
        # 2nd neigbours
        for idx_1st in connect_atom_idx_1sts:
            _connect_atom_idx_2nds = np.where(connect_map[idx_1st, :] > 3)[0]
            _connect_atom_idx_2nds = _connect_atom_idx_2nds[_connect_atom_idx_2nds != ii]
            connect_atom_idx_2nds = np.append(connect_atom_idx_2nds, _connect_atom_idx_2nds)

        # 2nd neighbour should not be direct neighbours
        filter_cond = [idx not in connect_atom_idx_1sts for idx in connect_atom_idx_2nds]
        connect_atom_idx_2nds = connect_atom_idx_2nds[filter_cond]
        if connect_atom_idx_2nds.shape[0] == 0:
            continue

        connect_atom_idx_2nds = np.unique(connect_atom_idx_2nds).astype(int)
        for idx_2nd in connect_atom_idx_2nds:
            smi_graph = update_virtual_bond_feature(smi_graph, ii, idx_2nd, -2)

        if num_neighbors >= 3 or ii in third_neighbours:
            continue

        # randomly choose a single 3rd neighbour
        random.shuffle(connect_atom_idx_2nds)
        for idx_2nd in connect_atom_idx_2nds:
            connect_atom_idx_3rds = np.where(connect_map[idx_2nd, :] > 3)[0]
            # 3rd neighbours should be neither 2nd or direct neighbours
            filter_idx = [idx not in connect_atom_idx_1sts and idx not in connect_atom_idx_2nds for idx in connect_atom_idx_3rds]
            connect_atom_idx_3rds = connect_atom_idx_3rds[filter_idx]
            if connect_atom_idx_3rds.shape[0] == 0:
                continue
            random.shuffle(connect_atom_idx_3rds)
            idx_3rd = connect_atom_idx_3rds[0]
            smi_graph = update_virtual_bond_feature(smi_graph, ii, idx_3rd, -1)
            third_neighbours.append(idx_3rd)
            break

    # mask_diag = np.ones((mol_len, mol_len))
    # np.fill_diagonal(mask_diag, 0)
    # mask_diag = np.expand_dims(mask_diag, axis=-1)
    # smi_graph *= mask_diag
    return smi_graph
