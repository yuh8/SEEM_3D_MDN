import numpy as np
from rdkit import DistanceGeometry
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdDistGeom

MAX_DISTANCE = 1E3
MIN_DISTANCE = 1E-3


def get_max_min_bound(bounds_matrix):
    num_atoms = bounds_matrix.shape[0]

    # Initial matrix
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if bounds_matrix[i, j] == 0:
                bounds_matrix[i, j] = MAX_DISTANCE
            if bounds_matrix[j, i] <= 0:
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


def embed_conformer(mol, means, stds, d_mean, d_std, mask, seed):
    num_atoms = len([atom.GetSymbol() for atom in mol.GetAtoms()])
    bound_upper = np.triu(means, 1) + np.triu(stds, 1)
    bound_upper *= d_std * np.triu(mask)
    bound_upper += d_mean * np.triu(mask)
    bound_upper = bound_upper[:num_atoms, :num_atoms]

    bound_lower = np.triu(means, 1) - np.triu(stds, 1)
    bound_lower *= d_std * np.triu(mask)
    bound_lower += d_mean * np.triu(mask)
    bound_lower = bound_lower[:num_atoms, :num_atoms].T
    bounds_matrix = bound_lower + bound_upper

    bounds_matrix = get_max_min_bound(bounds_matrix)
    np.fill_diagonal(bounds_matrix, 0)
    bounds_matrix = bounds_matrix.astype(np.double)

    return embed_bounds_matrix(mol, bounds_matrix, seed)


def embed_conformer_gt(mol, means, d_mean, d_std, mask, seed):
    num_atoms = len([atom.GetSymbol() for atom in mol.GetAtoms()])
    bound_upper = np.triu(means, 1) + 0.0000001
    bound_upper *= d_std * np.triu(mask)
    bound_upper += d_mean * np.triu(mask)
    bound_upper = np.triu(bound_upper[:num_atoms, :num_atoms])

    bound_lower = np.triu(means, 1) - 0.00000001
    bound_lower *= d_std * np.triu(mask)
    bound_lower += d_mean * np.triu(mask)
    bound_lower = np.triu(bound_lower[:num_atoms, :num_atoms]).T
    bounds_matrix = bound_lower + bound_upper

    bounds_matrix = get_max_min_bound(bounds_matrix)
    np.fill_diagonal(bounds_matrix, 0)
    bounds_matrix = bounds_matrix.astype(np.double)

    return embed_bounds_matrix(mol, bounds_matrix, seed)


def align_conformers(molecule, heavy_only=True):
    atom_ids = []
    if heavy_only:
        atom_ids = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    rdMolAlign.AlignMolConformers(molecule, atomIds=atom_ids)
