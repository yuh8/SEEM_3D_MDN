import numpy as np
from rdkit import DistanceGeometry
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdDistGeom
from scipy.special import softmax

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


def sample_bound_matrix(alpha, d_pred_mean, d_pred_std, num_atoms):
    bound_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i >= j:
                continue
            alpha_ij = softmax(alpha[i, j])
            alpha_idx = np.argmax(np.random.multinomial(1, alpha_ij))
            _mean = d_pred_mean[i, j, alpha_idx]
            _std = d_pred_std[i, j, alpha_idx]
            bound_matrix[i, j] = _mean + _std
            bound_matrix[j, i] = np.maximum(_mean - _std, MIN_DISTANCE)

    return bound_matrix


def embed_bounds_matrix(mol, bounds_matrix, num_confs, seed):
    DistanceGeometry.DoTriangleSmoothing(bounds_matrix)
    ps = rdDistGeom.EmbedParameters()
    ps.numThreads = 0  # max number of threads supported by the system will be used
    ps.useRandomCoords = True  # recommended for larger molecules
    ps.clearConfs = True
    ps.randomSeed = seed
    ps.SetBoundsMat(bounds_matrix)

    return rdDistGeom.EmbedMultipleConfs(mol, num_confs, ps)


def embed_conf(mol, bounds_matrix, seed):
    DistanceGeometry.DoTriangleSmoothing(bounds_matrix)
    ps = rdDistGeom.EmbedParameters()
    ps.numThreads = 0  # max number of threads supported by the system will be used
    ps.useRandomCoords = True  # recommended for larger molecules
    ps.clearConfs = False
    ps.randomSeed = seed
    ps.SetBoundsMat(bounds_matrix)

    return rdDistGeom.EmbedMolecule(mol, ps)


def embed_conformer(mol, num_confs, means, stds, d_mean, d_std, mask, seed):
    num_atoms = len([atom.GetSymbol() for atom in mol.GetAtoms()])
    bound_upper = np.triu(means, 1) + 3 * np.triu(stds, 1)
    bound_upper *= d_std * np.triu(mask)
    bound_upper += d_mean * np.triu(mask)
    bound_upper = bound_upper[:num_atoms, :num_atoms]

    bound_lower = np.triu(means, 1) - 3 * np.triu(stds, 1)
    bound_lower *= d_std * np.triu(mask)
    bound_lower += d_mean * np.triu(mask)
    bound_lower = bound_lower[:num_atoms, :num_atoms].T
    bounds_matrix = bound_lower + bound_upper

    bounds_matrix = get_max_min_bound(bounds_matrix)
    np.fill_diagonal(bounds_matrix, 0)
    bounds_matrix = bounds_matrix.astype(np.double)

    return embed_bounds_matrix(mol, bounds_matrix, num_confs, seed)


def align_conformers(molecule, heavy_only=True):
    atom_ids = []
    if heavy_only:
        atom_ids = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    rdMolAlign.AlignMolConformers(molecule, atomIds=atom_ids)
