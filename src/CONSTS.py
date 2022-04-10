import tensorflow as tf
from rdkit import Chem
TF_EPS = tf.keras.backend.epsilon()

# data gen
NUM_CONFS_PER_MOL = 5

# Bound features {Type:4, Stereo: 6}
BOND_DICT = Chem.rdchem.BondType.values
BOND_STEREO_DICT = Chem.rdchem.BondStereo.values
BOND_NAMES = list(BOND_DICT.values())[1:4]
BOND_NAMES.append(list(BOND_DICT.values())[12])
BOND_STEREO_NAMES = list(BOND_STEREO_DICT.values())


# Atom features
MAX_NUM_ATOMS = 160
ATOM_LIST = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'P', 'I', 'Na', 'B', 'Si', 'Se', 'K', 'Bi']
CHARGES = [-1, 0, 1, 2, 3]

HYBR_DICT = Chem.rdchem.HybridizationType.values
ATOM_HYBR_NAMES = list(HYBR_DICT.values())[1:7]

FEATURE_DEPTH = len(ATOM_LIST) + len(CHARGES) + len(ATOM_HYBR_NAMES) + len(BOND_NAMES) + len(BOND_STEREO_NAMES) + 1


# Mixture Gaussian
NUM_COMPS = 1
OUTPUT_DEPTH = NUM_COMPS + NUM_COMPS * 2


# train hps
BATCH_SIZE = 32
NUM_FILTERS = 128
FILTER_SIZE = 3
NUM_RES_BLOCKS = 20
BATCH_SIZE = 128
