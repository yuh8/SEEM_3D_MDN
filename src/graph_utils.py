import numpy as np
from networkx import Graph
from rdkit.Chem.rdchem import Mol
from typing import List


def mol_to_extended_graph(molecule: Mol, seed: int = 0) -> Graph:
    rng = np.random.default_rng(seed=seed)
    start = rng.integers(low=0, high=molecule.GetNumAtoms(), size=1).item()
    bond_graph = build_bond_graph(molecule)
    sequence = get_random_bf_sequence(graph=bond_graph, start=start, rng=rng)
    mol_len = len(sequence)

    graph = Graph()
    max_neighbor_len = -np.inf
    for new_node in sequence:
        neighbor_len = embed_node_in_graph(graph, new_node=new_node,
                                           bond_graph=bond_graph,
                                           max_dist=mol_len, rng=rng)
        if neighbor_len > max_neighbor_len:
            max_neighbor_len = neighbor_len

    return graph, max_neighbor_len


def build_bond_graph(molecule: Mol) -> Graph:
    graph = Graph()
    for bond in molecule.GetBonds():
        source_index, sink_index = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        graph.add_edge(source_index, sink_index)
    return graph


def embed_node_in_graph(graph, new_node: int, bond_graph: Graph, max_dist: int, rng: np.random.Generator) -> None:
    graph.add_node(new_node)
    bonded_neighborhood_list = get_neighborhoods(bond_graph, new_node, max_distance=max_dist)
    num_neighbors = sum([len(l) for l in bonded_neighborhood_list])
    assert max_dist == num_neighbors, 'missing neighbor'
    neighbor_len = len(bonded_neighborhood_list)

    for k in range(1, neighbor_len):
        neighborhood = bonded_neighborhood_list[k]
        rng.shuffle(neighborhood)

        for neighbor in neighborhood:
            if neighbor in graph.nodes and not graph.has_edge(new_node, neighbor):
                graph.add_edge(new_node, neighbor, kind=k)

    return neighbor_len


def get_neighborhoods(graph: Graph, source: int, max_distance: int) -> List[List[int]]:
    neighborhoods = [[source]]

    for k in range(1, max_distance + 1):
        new_neighborhood = []
        for front in neighborhoods[k - 1]:
            for neighbor in graph.neighbors(front):
                present = False
                for neighborhood in neighborhoods:
                    if neighbor in neighborhood or neighbor in new_neighborhood:
                        present = True
                        break

                if not present:
                    new_neighborhood.append(neighbor)

        if not new_neighborhood:
            break

        neighborhoods.append(new_neighborhood)

    return neighborhoods


def get_random_bf_sequence(graph: Graph, start: int, rng: np.random.Generator) -> List[int]:
    visited_list = [start]
    queue = [start]

    while len(queue):
        node = queue.pop(0)

        # shuffle neighbors
        neighbors = list(graph.neighbors(node))
        rng.shuffle(neighbors)

        for neighbor in neighbors:
            if neighbor not in visited_list:
                visited_list.append(neighbor)
                queue.append(neighbor)

    return visited_list
