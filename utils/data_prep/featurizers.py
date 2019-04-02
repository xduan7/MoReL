""" 
    File Name:          MoReL/featurizers.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               4/2/19
    Python Version:     3.5.4
    File Description:
"""
import re
import logging
import numpy as np
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

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

# High frequency/occurrence atoms
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
    # 'Se':   34,
    # 'K':    19,
    # 'Sn':   50,
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

# Tokenize numbers from ['0', ..., '16'] -> [144, 160]
NUMBER_TOKEN_DICT = {str(i): i + 144 for i in range(17)}

DEFAULT_TOKEN_DICT = {
    **SPECIAL_TOKEN_DICT,
    **ATOM_TOKEN_DICT,
    **NON_ATOM_TOKEN_DICT,
    **NUMBER_TOKEN_DICT,
}

# Graph features ##############################################################
ATOM_FEAT_FUNC_LIST = [
    # Numerical features
    Chem.Atom.GetAtomicNum,
    Chem.Atom.GetDegree,
    Chem.Atom.GetTotalNumHs,
    Chem.Atom.GetImplicitValence,
    Chem.Atom.GetIsAromatic,
    Chem.Atom.GetFormalCharge,
    Chem.Atom.GetNumRadicalElectrons,

    # TODO: one-hot encoding for the categorical features?
    Chem.Atom.GetHybridization,
    Chem.Atom.GetChiralTag,

    # Features that are not included in previous studies
    # Chem.Atom.GetExplicitValence,
    # Chem.Atom.GetTotalDegree,
    # Chem.Atom.IsInRing,
]

BOND_FEAT_FUNC_LIST = [
    # The first feature will always be bond existence
    # TODO: one-hot encoding
    Chem.Bond.GetBondType,
    Chem.Bond.GetIsConjugated,
    Chem.Bond.IsInRing,
    Chem.Bond.GetStereo,

    # Features that are not included in previous studies
    # Chem.Bond.GetBondDir,
    # Chem.Bond.GetIsAromatic,
]


# Featurization functions #####################################################
def inchi_to_mol(inchi: str) -> Optional[Chem.Mol]:
    try:
        mol: Chem.rdchem.Mol = Chem.MolFromInchi(inchi)
        assert mol is not None
    except AssertionError:
        logger.warning(f'Invalid InChI key: {inchi}')
        return None
    else:
        return mol


def mol_to_smiles(mol: Chem.Mol,
                  smiles_kwargs: dict = None) -> Optional[str]:
    try:
        assert mol
        smiles_kwargs = {} if smiles_kwargs is None else smiles_kwargs
        smiles = Chem.MolToSmiles(mol=mol, **smiles_kwargs)
    except AssertionError:
        logger.warning(f'Failed to convert Mol {mol} to SMILES')
        return None
    else:
        return smiles


def mol_to_tokens(mol: Chem.Mol,
                  len_tokens: int,
                  token_dict: dict = None,
                  smiles_kwargs: dict = None) -> Optional[np.array]:
    try:
        assert mol
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
                # Take care of the rare cases where there are double digits
                if ((i + 1) < len(smiles)) and smiles[i + 1].isdigit():
                    symbol = smiles[i: i + 2]
                    skip_next = True
                else:
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

        tokens = np.array(tokens, dtype=np.uint8)

    except AssertionError:
        logger.warning(f'Failed to convert Mol {mol} to tokens')
        return None
    else:
        return tokens


def mol_to_fingerprints(mol: Chem.Mol,
                        fp_kwargs: dict = None) -> Optional[np.array]:

    # TODO: Note that there are a lot of different fingerprint to try,
    #  but here we are only using ECFP, which is consistent with MoleculeNet
    # For more fingerprint, check outDIY Drug Discovery by Daniel C. Elton
    try:
        assert mol
        assert 'nBits' in fp_kwargs
        assert 'radius' in fp_kwargs
        fp_kwargs = {} if fp_kwargs is None else fp_kwargs
        fingerprints = AllChem.GetMorganFingerprint(mol=mol, **fp_kwargs)
    except AssertionError:
        logger.warning(f'Failed to convert Mol {mol} to fingerprints')
        return None
    else:
        return np.array(fingerprints, dtype=np.uint8)


def mol_to_descriptors(mol: Chem.Mol,
                       dscrptr_names: iter = None) -> Optional[np.array]:
    # Note that this function only converts molecules to 202 descriptors
    # implemented in RDkit
    try:
        assert mol
        descriptors = [func(mol) for name, func in Descriptors.descList
                       if (dscrptr_names is None) or (name in dscrptr_names)]
    except AssertionError:
        logger.warning(f'Failed to convert Mol {mol} to descriptors')
        return None
    else:
        return np.array(descriptors, dtype=np.float)


def mol_to_graph(mol: Chem.Mol,
                 ):

    """
    This implementation is based on:
        https://github.com/HIPS/neural-fingerprint/
        neuralfingerprint/mol_graph.py
    which is the git repo for https://arxiv.org/pdf/1509.09292.pdf

    And
        https://github.com/deepchem/deepchem/
        deepchem/feat/graph_features.py
    which is the git repo for DeepChem



    :param mol:
    :return:
    """






    pass







