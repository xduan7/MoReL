""" 
    File Name:          MoReL/featurizers.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               4/2/19
    Python Version:     3.5.4
    File Description:
"""
import re
import torch
import logging
import numpy as np
from typing import Optional, List, Dict
from itertools import product, combinations_with_replacement

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol
from rdkit.Chem.rdMolDescriptors import GetAtomPairFingerprint, \
    GetMACCSKeysFingerprint, GetMorganFingerprint, \
    GetTopologicalTorsionFingerprint
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity, DiceSimilarity, \
    CosineSimilarity, SokalSimilarity, RusselSimilarity, \
    RogotGoldbergSimilarity, AllBitSimilarity, KulczynskiSimilarity, \
    McConnaugheySimilarity, AsymmetricSimilarity, BraunBlanquetSimilarity, \
    TverskySimilarity

from torch_geometric.data import Data

# Suppress unnecessary RDkit warnings and errors
RDLogger.logger().setLevel(RDLogger.CRITICAL)
logger = logging.getLogger(__name__)

# Tokenization dictionaries ###################################################
# Special tokens for meta token
SPECIAL_TOKEN_DICT = {
    'SOS':  0,               # Start of the sentence
    'UNK':  128,             # Unknown atoms
    'MSK':  129,             # Masked tokens/atoms for prediction
    'EOS':  254,             # End of the sentence
    'PAD':  255,             # Padding
}

# High frequency/occurrence atoms from PCBA
ATOM_TOKEN_DICT = {
    'C':    6,
    'N':    7,
    'O':    8,
    'S':    16,
    'F':    9,
    'Cl':   17,
    'Br':   35,
    'P':    15,
    'I':    53,
    'Na':   11,
    'Si':   14,
    'B':    5,

    'Se':   34,
    'K':    19,
    'Sn':   50,

    'H':    1,
}

# Bonds and other structural characters
NON_ATOM_TOKEN_DICT = {
    # Bonds
    '.':    193,
    '-':    194,
    '=':    195,
    '#':    196,
    '$':    197,
    ':':    198,
    '/':    199,
    '\\':   200,

    # Annotations and charges
    '[':    224,
    ']':    225,
    '(':    226,
    ')':    227,
    '+':    228,
    '%':    229,
    '@':    230,
}

# Tokenize numbers from ['0', ..., '62'] -> [130, 192]
NUMBER_TOKEN_DICT = {str(i): i + 130 for i in range(63)}

DEFAULT_TOKEN_DICT = {
    **SPECIAL_TOKEN_DICT,
    **ATOM_TOKEN_DICT,
    **NON_ATOM_TOKEN_DICT,
    **NUMBER_TOKEN_DICT,
}

# Fingerprint features ########################################################
DEFAULT_FP_KWARGS = {
    'radius':   2,
    'nBits':    1024,
}

# Graph features ##############################################################
ATOM_FEAT_FUNC_DICT = {
    # http://www.rdkit.org/docs-beta/api/rdkit.Chem.rdchem.Atom-class.html
    # Numerical features
    'AtomicNum':            Chem.Atom.GetAtomicNum,
    'Degree':               Chem.Atom.GetDegree,
    'ExplicitValence':      Chem.Atom.GetExplicitValence,
    'FormalCharge':         Chem.Atom.GetFormalCharge,
    'ImplicitValence':      Chem.Atom.GetImplicitValence,
    'IsAromatic':           Chem.Atom.GetIsAromatic,
    'IsInRing':             Chem.Atom.IsInRing,
    'NumExplicitH':         Chem.Atom.GetNumExplicitHs,
    'NumImplicitHs':        Chem.Atom.GetNumImplicitHs,
    'NumRadicalElectrons':  Chem.Atom.GetNumRadicalElectrons,
    'TotalDegree':          Chem.Atom.GetTotalDegree,
    'TotalNumHs':           Chem.Atom.GetTotalNumHs,

    # Categorical features
    'ChiralTag':            Chem.Atom.GetChiralTag,
    'Hybridization':        Chem.Atom.GetHybridization,
}

DEFAULT_ATOM_FEAT_LIST = [
    'AtomicNum',
    'Degree',
    'TotalNumHs',
    'ImplicitValence',
    'IsAromatic',
    'Hybridization',
]

BOND_FEAT_FUNC_DICT = {
    # http://www.rdkit.org/docs-beta/api/rdkit.Chem.rdchem.Bond-class.html
    # Numerical features
    'IsAromatic':           Chem.Bond.GetIsAromatic,
    'IsConjugated':         Chem.Bond.GetIsConjugated,
    'IsInRing':             Chem.Bond.IsInRing,
    'ValenceContrib':       Chem.Bond.GetValenceContrib,

    # Categorical features
    'BondDir':              Chem.Bond.GetBondDir,
    'BondType':             Chem.Bond.GetBondType,
    'Stereo':               Chem.Bond.GetStereo,
}

DEFAULT_BOND_FEAT_LIST = [
    'BondType',
    'IsConjugated',
    'IsInRing',
]

# This dict saves all the possible values for categorical features for both
# atoms and bonds. Note that these should be commonly used values
DEFAULT_FEAT_VALUE_DICT = {
    'Hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'BondType': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    # TODO: fill out the possible values for ChiralTag, BondDir, Stereo
}


# Molecular distance features #################################################
FP_FUNC_DICT = {
    # https://www.rdkit.org/docs/GettingStartedInPython.html
    # https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html
    # Explicit fingerprint vector
    'MACCSKeys':        GetMACCSKeysFingerprint,
    'Topological':      FingerprintMol,

    # Sparse/implicit fingerprint vector
    'AtomPair':         GetAtomPairFingerprint,
    'Morgan':           GetMorganFingerprint,
    'Torsion':          GetTopologicalTorsionFingerprint,
}

DEFAULT_FP_FUNC_LIST = [
    'AtomPair',
    'Morgan',
    'Torsion',
]

DEFAULT_FP_FUNC_PARAM = {
    'MACCSKeys': [
        {}
    ],
    'Topological': [
        {}
    ],
    'AtomPair':[
        {'maxLength': 30},
        {'maxLength': 30, 'includeChirality': True},
    ],
    'Morgan': [
        {'radius': 2},
        {'radius': 3},
        {'radius': 2, 'useChirality': True, 'useFeatures': True},
        {'radius': 3, 'useChirality': True, 'useFeatures': True},
    ],
    'Torsion': [
        {'targetSize': 4},
        {'targetSize': 4, 'includeChirality': True},
    ],
}

SIM_FUNC_DICT = {
    # https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html
    'Tanimoto':             TanimotoSimilarity,
    'Dice':                 DiceSimilarity,
    'Cosine':               CosineSimilarity,
    'Sokal':                SokalSimilarity,
    'Russel':               RusselSimilarity,
    'RogotGoldberg':        RogotGoldbergSimilarity,
    'AllBit':               AllBitSimilarity,
    'Kulczynski':           KulczynskiSimilarity,
    'McConnaughey':         McConnaugheySimilarity,
    'Asymmetric':           AsymmetricSimilarity,
    'BraunBlanquet':        BraunBlanquetSimilarity,
    'TverskySimilarity':    TverskySimilarity,
}

DEFAULT_SIM_FUNC_LIST = [
    'Tanimoto',
    'Dice',
]


# Helper functions ############################################################
def one_hot_encode(value,
                   possible_values: List = None) -> List:
    """
    This function will one-hot encode a single value.
    Note that if possible values are not given, it will assume that the
    given values is of some atom/bond feature type in RDkit, and perform
    encoding accordingly.
    """

    if possible_values is None:
        possible_values = list(type(value).values.values())

    enc_feat = [0] * len(possible_values)
    try:
        enc_feat[possible_values.index(value)] = 1
    except ValueError:
        # Will return a vector of zeros if the feature is not listed in the
        # possible values
        pass

    return enc_feat


def mols_to_fp_list(mol_list: List[Chem.Mol],
                    fp_func_list: List[str] = None,
                    fp_func_param_dict: Dict[str, List] = None) -> List:

    mol_fp_list = []
    for mol in mol_list:

        __mol_fp = []
        for fp_func in fp_func_list:

            __fp_func: callable = FP_FUNC_DICT[fp_func]
            for fp_func_param in fp_func_param_dict[fp_func]:
                __mol_func_param_fp = __fp_func(mol, **fp_func_param)
                __mol_fp.append(__mol_func_param_fp)

        mol_fp_list.append(__mol_fp)

    return mol_fp_list


# Featurization functions #####################################################
def inchi_to_mol(inchi: str) -> Optional[Chem.Mol]:
    mol: Chem.Mol = Chem.MolFromInchi(inchi)
    if mol is None:
        logger.warning(f'Invalid InChI key: {inchi}')
    return mol


def mol_to_smiles(mol: Chem.Mol,
                  smiles_kwargs: dict = None) -> Optional[str]:

    smiles_kwargs = {} if smiles_kwargs is None else smiles_kwargs
    return Chem.MolToSmiles(mol=mol, **smiles_kwargs)


def mol_to_tokens(mol: Chem.Mol,
                  len_tokens: int = 128,
                  token_dict: dict = None,
                  smiles_kwargs: dict = None) -> Optional[torch.Tensor]:

    smiles = mol_to_smiles(mol, smiles_kwargs)
    token_dict = DEFAULT_TOKEN_DICT if token_dict is None else token_dict

    # Every token array starts with SOS
    tokens = [token_dict['SOS'], ]

    # Note that mol from smiles from mol will keep mol and smiles
    # consistent, which is important in tokenization
    # Note that this operation will take about several hundred us
    atom_list = [atom.GetSymbol()
                 for atom in Chem.MolFromSmiles(smiles).GetAtoms()]

    atom_index = 0
    skip_next = False
    for i, ci in enumerate(smiles):

        if skip_next:
            skip_next = False
            continue

        symbol = ''
        if ci.isalpha():
            next_atom = atom_list[atom_index] \
                if atom_index < len(atom_list) else ' '

            if bool(re.match(next_atom, smiles[i:], re.I)):
                skip_next = (len(next_atom) == 2)
                symbol = next_atom
                atom_index += 1
            else:
                # In this case, the only logical explanation is that i is a
                # hydrogen atom, which is completed ignored in atom list
                if ci.upper() == 'H':
                    symbol = 'H'
                else:
                    logger.warning(f'SMILES {smiles} is inconsistent')
                    return None

            if symbol not in token_dict:
                symbol = 'UNK'

        elif ci.isdigit():
            # # Take care of the rare cases where there are double digits
            # if ((i + 1) < len(smiles)) and smiles[i + 1].isdigit():
            #     symbol = smiles[i: i + 2]
            #     skip_next = True
            # else:
            #     symbol = ci
            symbol = ci

        elif not ci.isalnum():
            # Bonds, rings, etc.
            symbol = ci
            if symbol not in token_dict:
                print(f'Symbol {symbol} not in token dict')
                return None
            assert symbol in token_dict

        else:
            print(f'Unknown SMILES conversion at index {i} in {smiles}')

        tokens.append(token_dict[symbol])

    if len_tokens - len(tokens) > 0:
        tokens += [token_dict['PAD'], ] * (len_tokens - len(tokens))
    else:
        logger.warning(f'Tokens for {smiles} '
                       f'exceeds the given length {len_tokens}')
        return None
    return torch.from_numpy(np.array(tokens, dtype=np.float32))


def mol_to_fingerprints(mol: Chem.Mol,
                        fp_kwargs: dict = None) -> Optional[torch.Tensor]:

    # TODO: Note that there are a lot of different fingerprint to try,
    #  but here we are only using ECFP, which is consistent with MoleculeNet
    # For more fingerprint, check outDIY Drug Discovery by Daniel C. Elton
    fp_kwargs = DEFAULT_FP_KWARGS if fp_kwargs is None else fp_kwargs
    fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol=mol, **fp_kwargs)
    return torch.from_numpy(np.array(fingerprints, dtype=np.float32))


def mol_to_descriptors(mol: Chem.Mol,
                       dscrptr_names: iter = None) -> Optional[torch.Tensor]:
    # Note that this function only converts molecules to 202 descriptors
    # implemented in RDkit
    descriptors = [func(mol) for name, func in Descriptors.descList
                   if (dscrptr_names is None) or (name in dscrptr_names)]
    return torch.from_numpy(np.array(descriptors, dtype=np.float32))


def mol_to_graph(mol: Chem.Mol,
                 master_atom: bool = True,
                 master_bond: bool = True,
                 max_num_atoms: int = -1,
                 atom_feat_list: List[str] = None,
                 bond_feat_list: List[str] = None) -> Optional[Data]:

    """
    This implementation is based on:
        https://github.com/HIPS/neural-fingerprint/
        neuralfingerprint/features.py
    which is the git repo for https://arxiv.org/pdf/1509.09292.pdf

    And
        https://github.com/deepchem/deepchem/
        deepchem/feat/graph_features.py
    which is the git repo for DeepChem
    """

    # Sanity check for graph size
    num_atoms = mol.GetNumAtoms() + master_atom
    if (num_atoms > max_num_atoms) and (max_num_atoms >=0):
        logger.warning(f'Number of atoms for {Chem.MolToSmiles(mol)} '
                       f'exceeds the maximum number of atoms {max_num_atoms}')
        return None

    if atom_feat_list is None:
        atom_feat_list = DEFAULT_ATOM_FEAT_LIST
    if bond_feat_list is None:
        bond_feat_list = DEFAULT_BOND_FEAT_LIST

    # Process the graph in the way that aligns with PyG
    # Returning (node_attr=[N, F], edge_index=[2, M], edge_attr=[M, E])
    # TODO: add position information for the atoms?

    # Prepare features for atoms/nodes
    node_attr = []
    for atom in mol.GetAtoms():

        single_node_attr = []
        for atom_feat_name in atom_feat_list:

            feat_func: callable = ATOM_FEAT_FUNC_DICT[atom_feat_name]
            feat = feat_func(atom)

            # If the feature is not numeric, then one-hot encoding
            if type(feat) in [int, float, bool]:
                single_node_attr.append(feat)
            else:
                possible_values = DEFAULT_FEAT_VALUE_DICT[atom_feat_name] \
                    if atom_feat_name in DEFAULT_FEAT_VALUE_DICT else None
                attr = one_hot_encode(feat, possible_values)
                single_node_attr.extend(list(attr))

        node_attr.append(single_node_attr)

    # Prepare features for bonds/edges
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():

        # edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        single_edge_attr = [1, ] if master_bond else []
        for bond_feat_name in bond_feat_list:

            feat_func: callable = BOND_FEAT_FUNC_DICT[bond_feat_name]
            feat = feat_func(bond)

            # If the feature is not numeric, then one-hot encoding
            if type(feat) in [int, float, bool]:
                single_edge_attr.append(feat)
            else:
                possible_values = DEFAULT_FEAT_VALUE_DICT[bond_feat_name] \
                    if bond_feat_name in DEFAULT_FEAT_VALUE_DICT else None
                attr = one_hot_encode(feat, possible_values)
                single_edge_attr.extend(list(attr))

        # Note that in molecules, bonds are always mutually shared
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_attr.append(single_edge_attr)
        edge_attr.append(single_edge_attr)

    node_attr = np.array(node_attr, dtype=np.float32)
    edge_index = np.transpose(np.array(edge_index, dtype=np.int64))
    edge_attr = np.array(edge_attr, dtype=np.float32)

    return Data(x=torch.from_numpy(node_attr),
                edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_attr))


# TODO: mol_to_image, mol_to_jtnn
# Note that MolToImage is already implemented in RDkit


def mols_to_simmat(mol_list: List[Chem.Mol],
                   ref_mol_list: List[Chem.Mol] = None,
                   fp_func_list: List[str] = None,
                   fp_func_param_dict: Dict[str, List] = None,
                   sim_func_list: List[str] = None) -> np.array:

    if ref_mol_list is None:
        ref_mol_list = mol_list
        __self_ref = True
    else:
        __self_ref = False

    if fp_func_list is None:
        fp_func_list = DEFAULT_FP_FUNC_LIST

    if fp_func_param_dict is None:
        fp_func_param_dict = DEFAULT_FP_FUNC_PARAM
    else:
        fp_func_param_dict = {**DEFAULT_FP_FUNC_PARAM, **fp_func_param_dict}

    if sim_func_list is None:
        sim_func_list = DEFAULT_SIM_FUNC_LIST

    # Calculate all the fingerprints for all the molecules
    mol_fp_list = []
    for mol in mol_list:

        __mol_fp = []
        for fp_func in fp_func_list:

            __fp_func: callable = FP_FUNC_DICT[fp_func]
            for fp_func_param in fp_func_param_dict[fp_func]:
                __mol_func_param_fp = __fp_func(mol, **fp_func_param)
                __mol_fp.append(__mol_func_param_fp)

        mol_fp_list.append(__mol_fp)

    mol_fp_list = mols_to_fp_list(mol_list,
                                  fp_func_list,
                                  fp_func_param_dict)

    if __self_ref:
        ref_mol_fp_list = mol_fp_list
    else:
        ref_mol_fp_list = mols_to_fp_list(ref_mol_list,
                                          fp_func_list,
                                          fp_func_param_dict)

    # Similarity matrix
    simmat = np.zeros(shape=(len(mol_list), len(ref_mol_list),
                             len(mol_fp_list[0]) * len(sim_func_list)),
                      dtype=np.float32)

    # Keep a list of boolean that keeps track of which similarity is of valid
    sim_index_indicator = [True] * (len(mol_fp_list[0]) * len(sim_func_list))

    iterations = combinations_with_replacement(range(len(mol_list)), 2) \
        if __self_ref else \
        product(range(len(mol_list)), range(len(ref_mol_list)))

    for i, j in iterations:

        __fp_i, __fp_j = mol_fp_list[i], ref_mol_fp_list[j]

        for k, (__fp_i_k, __fp_j_k) in enumerate(zip(__fp_i, __fp_j)):

            for l, sim_func_l in enumerate(sim_func_list):

                __sim_func_l = SIM_FUNC_DICT[sim_func_l]
                __sim_index = k * len(sim_func_list) + l

                try:
                    assert sim_index_indicator[__sim_index]
                    __sim_i_j_k_l = __sim_func_l(__fp_i_k, __fp_j_k)
                except:
                    simmat[i, j, __sim_index] = np.nan
                    if __self_ref:
                        simmat[j, i, __sim_index] = np.nan
                    sim_index_indicator[__sim_index] = False
                    continue

                simmat[i, j, __sim_index] = __sim_i_j_k_l
                if __self_ref:
                    simmat[j, i, __sim_index] = __sim_i_j_k_l

    return simmat[:, :, sim_index_indicator]


if __name__ == '__main__':

    # Example SMILES from daylight.com website
    example_smiles_list = [
        'CCc1nn(C)c2c(=O)[nH]c(nc12)c3cc(ccc3OCC)S(=O)(=O)N4CCN(C)CC4',
        'Cc1nnc2CN=C(c3ccccc3)c4cc(Cl)ccc4-n12',
        'CC(C)(N)Cc1ccccc1',
        'CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13',
        'CN(C)C(=O)Cc1c(nc2ccc(C)cn12)c3ccc(C)cc3',
        'COc1ccc2[nH]c(nc2c1)S(=O)Cc3ncc(C)c(OC)c3C',
        'CS(=O)(=O)c1ccc(cc1)C2=C(C(=O)OC2)c3ccccc3',
        'Fc1ccc(cc1)C2CCNCC2COc3ccc4OCOc4c3',
        'CC(C)c1c(C(=O)Nc2ccccc2)c(c(c3ccc(F)cc3)n1CC[C@@H]4C[C@@H](O)CC('
        '=O)O4)c5ccccc5',
        'CN1CC(=O)N2[C@@H](c3[nH]c4ccccc4c3C[C@@H]2C1=O)c5ccc6OCOc6c5',
        'O=C1C[C@H]2OCC=C3CN4CC[C@@]56[C@H]4C[C@H]3[C@H]2[C@H]6N1c7ccccc75',
        'COC(=O)[C@H]1[C@@H]2CC[C@H](C[C@@H]1OC(=O)c3ccccc3)N2C',
        'COc1ccc2nccc([C@@H](O)[C@H]3C[C@@H]4CCN3C[C@@H]4C=C)c2c1',
        'CN1C[C@@H](C=C2[C@H]1Cc3c[nH]c4cccc2c34)C(=O)O',
        'CCN(CC)C(=O)[C@H]1CN(C)[C@@H]2Cc3c[nH]c4cccc(C2=C1)c34',
        'CN1CC[C@]23[C@H]4Oc5c3c(C[C@@H]1[C@@H]2C=C[C@@H]4O)ccc5O',
        'CN1CC[C@]23[C@H]4Oc5c3c(C[C@@H]1[C@@H]2C=C[C@@H]4OC(=O)C)ccc5OC(=O)C',
        'CN1CCC[C@H]1c2cccnc2',
        'Cn1cnc2n(C)c(=O)n(C)c(=O)c12',
        'C/C(=C\\CO)/C=C/C=C(/C)\\C=C\\C1=C(C)CCCC1(C)C',
    ]

    for s in example_smiles_list:

        m: Chem.Mol = Chem.MolFromSmiles(s)

        t = mol_to_tokens(m, 64)
        fp = mol_to_fingerprints(m)
        d = mol_to_descriptors(m)

        g = mol_to_graph(m, True, True, 128)
        print(g)
        # assert n.shape[0] == m.GetNumAtoms()
        # assert adj.shape[1] == e.shape[0]
        # Convert edge attributes to adjacency matrix
        # tmp = torch.masked_select(adj, mask=e[:, 2].byte()).view(2, -1)

    # Test molecular similarity matrix generation
    mol_list = [Chem.MolFromSmiles(s) for s in example_smiles_list]
    simmat = mols_to_simmat(mol_list,
                            fp_func_list=list(FP_FUNC_DICT.keys()),
                            sim_func_list=list(SIM_FUNC_DICT.keys()))

