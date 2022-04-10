import numpy as np
from rdkit import DistanceGeometry
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdDistGeom

MAX_DISTANCE = 1E3
MIN_DISTANCE = 1E-3


def get_init_bounds_matrix(mol):
    num_atoms = mol.GetNumAtoms()
    bounds_matrix = np.zeros(shape=(num_atoms, num_atoms), dtype=np.float)

    # Initial matrix
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            bounds_matrix[i, j] = MAX_DISTANCE
            bounds_matrix[j, i] = MIN_DISTANCE

    return bounds_matrix


def embed_bounds_matrix(mol, bounds_matrix, seed):
    DistanceGeometry.DoTriangleSmoothing(bounds_matrix)

    ps = rdDistGeom.EmbedParameters()
    ps.numThreads = 0  # max number of threads supported by the system will be used
    ps.useRandomCoords = True  # recommended for larger molecules
    ps.clearConfs = False
    ps.randomSeed = seed
    ps.SetBoundsMat(bounds_matrix)

    return rdDistGeom.EmbedMolecule(mol, ps)


def embed_conformer(mol, means, stds, seed):
    bounds_matrix = get_init_bounds_matrix(mol)
    num_atoms = len([atom.GetSymbol() for atom in mol.GetAtoms()])
    bound_upper = np.triu(means, 1) + np.triu(stds, 1)
    bound_lower = np.triu(means, 1) - np.triu(stds, 1)
    bounds_matrix = np.triu(bound_upper, 1) + np.triu(bound_lower, 1).T
    bounds_matrix = bounds_matrix[:num_atoms, :num_atoms]
    bounds_matrix[bounds_matrix < 0] = MIN_DISTANCE
    bounds_matrix = np.double(bounds_matrix)
    return embed_bounds_matrix(mol, bounds_matrix, seed)


def align_conformers(molecule, heavy_only=True):
    atom_ids = []
    if heavy_only:
        atom_ids = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    rdMolAlign.AlignMolConformers(molecule, atomIds=atom_ids)
