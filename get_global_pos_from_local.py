import numpy as np
from networkx import Graph


def get_neighbour_info(mol):
    bond_graph = build_bond_graph(mol)
    new_sequence = [i for i in range(len(mol.GetAtoms()))]
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    elements = [elements[i] for i in new_sequence]
    num_atoms = len(elements)

    # check validity of the molecule
    if not check_chain_connection(mol, new_sequence, bond_graph, 3):
        return None

    # get root nodes for the molecule tree
    root_nodes = [new_sequence[0]] + get_node_neighbours(mol, new_sequence[0], bond_graph, 3)

    # build the tree
    avail_nodes = [i for i in new_sequence if i not in root_nodes]
    positioned_nodes = root_nodes.copy()
    neighbour_index_dict = {0: root_nodes}
    neighbour_index_dict.update({new_sequence.index(root_nodes[-1]): [root_nodes[-1]] + root_nodes[:-1][::-1]})
    while avail_nodes:
        for avail_node in avail_nodes:
            for positioned_node in positioned_nodes:
                if mol.GetBondBetweenAtoms(avail_node, positioned_node) is not None:
                    if positioned_node in root_nodes:
                        root_node_index = root_nodes.index(positioned_node)
                        if root_node_index < 2:
                            neighbour_index_dict.update({new_sequence.index(avail_node): [avail_node] + root_nodes[root_node_index:root_node_index + 3]})
                        else:
                            neighbour_index_dict.update({new_sequence.index(avail_node): [avail_node] + root_nodes[root_node_index - 2:root_node_index + 1][::-1]})
                    else:
                        positioned_node_index = new_sequence.index(positioned_node)
                        neighbours = neighbour_index_dict[positioned_node_index]
                        neighbour_index_dict.update({new_sequence.index(avail_node): [avail_node] + neighbours[:3]})

                    avail_nodes.remove(avail_node)
                    positioned_nodes.append(avail_node)
                    break
    return neighbour_index_dict, positioned_nodes, new_sequence, num_atoms


def get_global_pos(coords_l, positioned_nodes, num_atoms, new_sequence, neighbour_index_dict):
    root_nodes = neighbour_index_dict[0]

    # define global axes
    x_g = np.array([1., 0., 0.])
    y_g = np.array([0., 1., 0.])
    z_g = np.array([0., 0., 1.])
    global_axes = np.array([x_g, y_g, z_g]).astype(np.float32)
    # find global coordinates for root nodes
    nodes_pos = {}
    d01 = coords_l[0][0]
    d12 = coords_l[0][1]
    first_bond_angle = coords_l[0][2]
    for i, root_node in enumerate(root_nodes):
        # first node positioned at global coordinates (0.,0.,0)
        if i == 0:
            nodes_pos.update({root_node: np.array([0., 0., 0.]).astype(np.float32)})
        # second node aligned to the global x axis with a predicetd distance
        elif i == 1:
            nodes_pos.update({root_node: np.array([d01, 0., 0.]).astype(np.float32)})
        # third node pos calculated based on predicted bond angle with respect to the
        # first two nodes and the predicted bond angle between first three nodes
        elif i == 2:
            delta_x = d12 * np.cos(first_bond_angle)
            delta_y = d12 * np.sin(first_bond_angle)
            if first_bond_angle > 0.5 * np.pi:
                nodes_pos.update({root_node: np.array([d01 - delta_x, delta_y, 0.]).astype(np.float32)})
            else:
                nodes_pos.update({root_node: np.array([d01 + delta_x, delta_y, 0.]).astype(np.float32)})
        # last node pos calculated from spherical coordinates based on the first three nodes
        else:
            node_index = new_sequence.index(root_node)
            # get local cartesian coordinates
            root_local_coords_c = coords_l[node_index]

            # define local axes
            x_l, y_l, z_l = get_principal_axes(
                nodes_pos[root_nodes[-2]],
                nodes_pos[root_nodes[-3]],
                nodes_pos[root_nodes[-4]]
            )
            local_axes = np.array([x_l, y_l, z_l]).astype(np.float32)

            # transformation matrix between local and global
            l2g = np.dot(global_axes, np.linalg.pinv(local_axes))
            t = nodes_pos[root_nodes[-2]]

            # transform local cartesian coordinates to global
            root_global_coords = np.array(np.dot(l2g, root_local_coords_c) + t).astype(np.float32)
            nodes_pos.update({root_node: root_global_coords})

    for avail_node in positioned_nodes[4:]:
        avail_node_idx = new_sequence.index(avail_node)
        avail_node_neighbours = neighbour_index_dict[avail_node_idx][1:]
        positioned_node = avail_node_neighbours[0]
        current_origin_pos = nodes_pos[positioned_node]
        avail_node_l_coords_c = coords_l[avail_node_idx]
        if positioned_node in root_nodes:
            root_node_index = root_nodes.index(positioned_node)
            if root_node_index < 2:
                neighbour_1_pos = nodes_pos[root_nodes[root_node_index + 1]]
                neighbour_2_pos = nodes_pos[root_nodes[root_node_index + 2]]
            else:
                neighbour_1_pos = nodes_pos[root_nodes[root_node_index - 1]]
                neighbour_2_pos = nodes_pos[root_nodes[root_node_index - 2]]
        else:
            positioned_node_index = new_sequence.index(positioned_node)
            neighbours = neighbour_index_dict[positioned_node_index][:3]
            neighbour_1_pos = nodes_pos[neighbours[1]]
            neighbour_2_pos = nodes_pos[neighbours[2]]
        # build local coordinate system
        x_l, y_l, z_l = get_principal_axes(
            current_origin_pos,
            neighbour_1_pos,
            neighbour_2_pos
        )
        local_axes = np.array([x_l, y_l, z_l]).astype(np.float32)
        # transformation matrix between local and global
        l2g = np.dot(global_axes, np.linalg.pinv(local_axes))
        t = current_origin_pos
        # transform local cartesian coordinates to global
        avail_node_global_coords = np.array(np.dot(l2g, avail_node_l_coords_c) + t).astype(np.float32)
        nodes_pos.update({avail_node: avail_node_global_coords})

    new_pos = []
    for i in range(num_atoms):
        new_pos.append(nodes_pos[i])
    new_pos = np.array(new_pos)

    return new_pos


def build_bond_graph(molecule):
    graph = Graph()
    for bond in molecule.GetBonds():
        source_index, sink_index = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        graph.add_edge(source_index, sink_index)
    return graph


def check_chain_connection(mol, sequence, bond_graph, num_hop):
    valid = True
    for new_node in sequence:
        neighbour_list = get_node_neighbours(mol, new_node, bond_graph, num_hop)
        if neighbour_list is None:
            valid = False
            break
    return valid


def get_node_neighbours(mol, new_node, bond_graph, num_hop):
    neighbour_lists = get_neighbourhoods(bond_graph, new_node, num_hop)
    # filter out atoms only having two chain neighbour because this is not possible
    if all(len(neighbours) > 0 for neighbours in neighbour_lists):
        neighbour_list = get_chain_connections(mol, neighbour_lists[1], neighbour_lists[2], neighbour_lists[3])
        assert mol.GetBondBetweenAtoms(neighbour_list[0], neighbour_list[1]) is not None, 'second hop connection faulty!'
        assert mol.GetBondBetweenAtoms(neighbour_list[1], neighbour_list[2]) is not None, 'third hop connection faulty!'
    else:
        return None

    return neighbour_list


def get_neighbourhoods(graph, source, num_hop):
    neighborhoods = [[source]]

    for k in range(1, num_hop + 1):
        new_neighborhood = []
        for front in neighborhoods[k - 1]:
            for neighbor in graph.neighbors(front):
                present = False
                for neighborhood in neighborhoods:
                    if neighbor in neighborhood:
                        present = True
                        break

                if not present:
                    new_neighborhood.append(neighbor)

        neighborhoods.append(new_neighborhood)
    return neighborhoods


def get_chain_connections(mol, neighours1, neighbours2, neighbours3):
    neighbour_chain = []
    for n1 in neighours1:
        for n2 in neighbours2:
            if mol.GetBondBetweenAtoms(n1, n2) is not None:
                for n3 in neighbours3:
                    if mol.GetBondBetweenAtoms(n2, n3) is not None:
                        neighbour_chain = [n1, n2, n3]
    return neighbour_chain


def get_principal_axes(origin, neighbour1, neighbour2):
    plane_vector_1 = neighbour1 - origin
    plane_vector_2 = neighbour2 - origin
    z = normal_from_vectors(plane_vector_1, plane_vector_2)
    z = z / np.linalg.norm(z)
    y = neighbour2 - origin
    y /= np.linalg.norm(y)
    x = normal_from_vectors(y, z)

    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def normal_from_vectors(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)

    return np.cross(unit_vector_1, unit_vector_2)


if __name__ == '__main__':
    # provide reference mol file and predicted local coordinates
    mol = 'RDKIT mol object'
    neighbour_index_dict, positioned_nodes, new_sequence, num_atoms = get_neighbour_info(mol)
    coords_l = '(MAX_NUM_ATOMS, 3)'
    new_pos = get_global_pos(coords_l, positioned_nodes, num_atoms, new_sequence, neighbour_index_dict)
