""" 
    File Name:          MoReL/smiles_prep.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/11/19
    Python Version:     3.5.4
    File Description:   

"""
import numpy as np
from rdkit import Chem
import utils.data_prep.config as c


def mol_to_smiles(mol: Chem.rdchem.Mol) -> str:

    return Chem.MolToSmiles(
        mol=mol,
        allBondsExplicit=c.ALL_BONDS_EXPLICIT,
        allHsExplicit=c.ALL_HS_EXPLICIT)


def tokenize_smiles(smiles: str,
                    padding: bool = c.SMILES_PADDING,
                    token_dict: dict = c.SMILES_TOKEN_DICT,
                    num_tokens: int = c.MAX_LEN_TOKENIZED_SMILES) -> np.array:
    """
    This function takes a SMILES string and a list of common atoms and
    performs tokenization. Each Atom and bound will be tokenized into an
    integer number (np.uint8).

    No sanity check required.

    :param smiles:
    :param padding:
    :param token_dict:
    :param num_tokens: length of token array (including SOS, EOS, UNK, and PAD)

    :return:
    """

    mol: Chem.Mol = Chem.MolFromSmiles(smiles)
    atom_list: list = mol.GetAtoms()

    assert len(smiles) < num_tokens - 1

    tokens = [token_dict['SOS'], ]

    skip_next = False
    for i, ci in enumerate(smiles):

        if skip_next:
            skip_next = False
            continue

        symbol = ''

        if ci.isalpha():
            pass
        # TODO: should do it with the help of GetAtoms()
            # cj = smiles[i + 1]
            # if ((i + 1) < len(smiles)) and cj.islower():
            #
            #     if (ci.upper() + cj) in token_dict:
            #         symbol = ci.upper() + cj
            #     elif ci.upper() in token_dict:
            #         symbol = ci.upper()
            #
            # else:
            #     symbol = ci.upper()

        # if ci.isupper():
        #     # Leading character of a molecule
        #     if ((i + 1) < len(smiles)) and smiles[i + 1].islower():
        #         symbol = smiles[i: i + 2]
        #         skip_next = True
        #     else:
        #         symbol = ci
        #
        # elif ci.islower():
        #     if ci.upper() in token_dict:
        #         symbol = ci.upper()
        #     else:
        #         print('Unconverted lower case char at index %i in %s'
        #               % (i,  smiles))

        elif ci.isdigit():
            if ((i + 1) < len(smiles)) and smiles[i + 1].isdigit():
                symbol = smiles[i: i + 2]
                skip_next = True
            else:
                symbol = ci

        elif not ci.isalnum():
            # Bonds, rings, etc.
            symbol = ci
            # Make sure
            if symbol not in token_dict:
                print(symbol)
            assert symbol in token_dict

        else:
            print('Unknown SMILES conversion at index %i in %s' % (i, smiles))

        print(symbol)

        # Get the corresponding tokens
        if symbol in token_dict:
            tokens.append(token_dict[symbol])
        else:
            tokens.append(token_dict['UNK'])

    if padding:
        tokens += [token_dict['PAD'], ] * (num_tokens - len(tokens))

    return np.array(tokens, dtype=np.uint8)


if __name__ == '__main__':

    # This should be tokenized into Cl, Br, UNK, C, C, [, UNK, ]
    # smiles1 = 'ClBrBaCC[D]'
    smiles1 = 'O=[N+]([O-])c1ccc(Cl)c([N+](=O)[O-])c1'

    mol1 = Chem.MolFromSmiles(smiles1)
    print(mol1.GetAtoms())


    for a in mol1.GetAtoms():
        print(a.GetSymbol())
    # print(Chem.MolToSmiles())

    tokens1 = tokenize_smiles(smiles1, padding=False)
    print(tokens1)
    print(len(tokens1))
