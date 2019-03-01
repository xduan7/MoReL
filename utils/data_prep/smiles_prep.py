""" 
    File Name:          MoReL/smiles_prep.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/11/19
    Python Version:     3.5.4
    File Description:   

"""
import re
import numpy as np
from rdkit import Chem
import utils.data_prep.config as c


def mol_to_smiles(mol: Chem.rdchem.Mol) -> str:
    try:
        return Chem.MolToSmiles(
            mol=mol,
            allBondsExplicit=c.ALL_BONDS_EXPLICIT,
            allHsExplicit=c.ALL_HS_EXPLICIT)
    except:
        return None


# def smiles_to_token(smiles: str,
#                     padding: bool = c.SMILES_PADDING,
#                     token_dict: dict = c.SMILES_TOKEN_DICT,
#                     num_tokens: int = c.MAX_LEN_TOKENIZED_SMILES)
#                     -> np.array:
#     """
#     Deprecated
#
#     This function takes a SMILES string and a list of common atoms and
#     performs tokenization. Each Atom and bound will be tokenized into an
#     integer number (np.uint8).
#
#     No sanity check required.
#
#     :param smiles:
#     :param padding:
#     :param token_dict:
#     :param num_tokens: length of token array
#                        (including SOS, EOS, UNK, and PAD)
#
#     :return:
#     """
#     assert len(smiles) < num_tokens - 1
#
#     # Every token array starts with SOS
#     tokens = [token_dict['SOS'], ]
#
#     skip_next = False
#     for i, ci in enumerate(smiles):
#
#         if skip_next:
#             skip_next = False
#             continue
#
#         symbol = ''
#
#         if ci.isalpha():
#             # Note that this part is merely a guess, no actual guarantee on
#             # the correctness of tokenization of SMILES strings
#             if (i + 1) < len(smiles) and smiles[i + 1].islower():
#                 cj = smiles[i + 1]
#                 if (ci.upper() + cj) in token_dict:
#                     symbol = ci.upper() + cj
#                     skip_next = True
#                 elif ci.upper() in token_dict:
#                     symbol = ci.upper()
#                 else:
#                     # Here we have no idea where [ci, cj] is a single atom
#                     # or not. We can only Guess
#                     if ci.isupper():
#                         symbol = ci.upper() + cj
#                     else:
#                         symbol = ci.upper()
#             else:
#                 symbol = ci.upper()
#
#         elif ci.isdigit():
#             if ((i + 1) < len(smiles)) and smiles[i + 1].isdigit():
#                 symbol = smiles[i: i + 2]
#                 skip_next = True
#             else:
#                 symbol = ci
#
#         elif not ci.isalnum():
#             # Bonds, rings, etc.
#             symbol = ci
#             # Make sure
#             if symbol not in token_dict:
#                 print(symbol)
#             assert symbol in token_dict
#
#         else:
#             print('Unknown SMILES conversion at index %i in %s'
#                   % (i, smiles))
#
#         # Get the corresponding tokens
#         if symbol in token_dict:
#             tokens.append(token_dict[symbol])
#         else:
#             tokens.append(token_dict['UNK'])
#
#     if padding:
#         tokens += [token_dict['PAD'], ] * (num_tokens - len(tokens))
#
#     return np.array(tokens, dtype=np.uint8)


def mol_to_token(mol: Chem.Mol,
                 padding: bool = c.SMILES_PADDING,
                 token_dict: dict = c.SMILES_TOKEN_DICT,
                 num_tokens: int = c.MAX_LEN_TOKENIZED_SMILES) -> np.array:
    """
    This function takes a different approach from smiles_to_tokens
    No sanity check required.

    :param mol:
    :param padding:
    :param token_dict:
    :param num_tokens: length of token array (including SOS, EOS, UNK, and PAD)

    :return:
    """
    # Every token array starts with SOS
    tokens = [token_dict['SOS'], ]

    smiles = mol_to_smiles(mol)
    if (mol is None) or (smiles is None):
        return None

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
        # print('parsing the %i th symbol %s ' % (i, ci))
        if ci.isalpha():
            next_atom = atom_list[atom_index] \
                if atom_index < len(atom_list) else ' '

            # print('processing %s (next atom = %s)...' % (ci, next_atom))

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
                    print('SMILES %s is inconsistent with atom list %s'
                          % (smiles, ''.join(atom_list)))
                    return None

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
                print('Symbol %s not in token dict' % symbol)
                return None
            assert symbol in token_dict

        else:
            print('Unknown SMILES conversion at index %i in %s' % (i, smiles))

        # Get the corresponding tokens
        # print('Confirming symbol %s' % symbol)
        if symbol in token_dict:
            tokens.append(token_dict[symbol])
        else:
            tokens.append(token_dict['UNK'])

    if padding and (num_tokens - len(tokens) > 0):
        tokens += [token_dict['PAD'], ] * (num_tokens - len(tokens))

    return np.array(tokens, dtype=np.uint8) \
        if len(tokens) <= num_tokens else None


if __name__ == '__main__':

    # Some examples demonstrating the difference between
    # smiles_to_token and mol_to_token
    # smiles1 = 'CCCCc1[se]c(N=C(O)c2ccc(OC)c(OC)c2)nc1-c1ccc(OC)cc1'
    # smiles1 = '[As].[In]'
    # smiles1 = 'O=[N+]([O-])O.[In]'
    # smiles1 = 'C1CCC(C2CCCC[N-]2)[N-]C1.CC(C)C([NH-])C(=O)[O-].Cl.[Pt+]'

    # The following three cases are shown to be inconsistent in the way that
    # Mol from InChI has different atom ordering from Mol from SMILES
    # smiles1 = 'C1=NC=c2c(ccc[n+]2CCCCC[n+]2ccccc2-c2ccccn2)=C/C=C/1.[Br-]'
    # smiles1 = 'C=Cc1c2[n-]c(c1C)C=c1[n-]c(c(CCC(=O)[O-])c1C)=C1c3[nH]c(c(' \
    #           'C)c3C(=O)C1C(=O)[O-])C=c1[n-]c(c(C)c1CC)=C2.[Cu].[Na+].[Na+]'
    smiles1 = 'Cc1c2[n-]c(c1CCC(=O)[O-])C=c1[nH]c(c(C)c1CCC(=O)[O-])=Cc1[' \
              'nH]c(c(C(C)O)c1C)C=c1[n-]c(c(C(C)O)c1C)=C2.Cl.Cl.[Pt+4]'
    mol1 = Chem.MolFromSmiles(smiles1)
    assert smiles1 == Chem.MolToSmiles(mol1)
    print([a.GetSymbol() for a in mol1.GetAtoms()])
    # print(smiles_to_token(smiles1, padding=False))
    print(mol_to_token(mol1, padding=False))

    # # Load more smiles to experiment in terms of speed and correctness
    # # between smiles_to_token and mol_to_token
    # import pandas as pd
    # from time import time
    # from os.path import join
    #
    # # Suppress warnings and errors from RDkit
    # from rdkit import RDLogger
    # RDLogger.logger().setLevel(RDLogger.CRITICAL)
    #
    # test_size = 2 ** 16
    #
    # cid_smiles_df = pd.read_csv(join(
    #     c.PROCESSED_DATA_DIR, 'CID-SMILES(PCBA).csv')).sample(test_size)
    #
    # smiles_list = cid_smiles_df['SMILES'].tolist()
    # mol_list = [Chem.MolFromSmiles(s) for s in smiles_list]
    #
    # assert all([(m is not None) for m in mol_list])
    #
    # start_time = time()
    # token_list1 = [mol_to_token(m) for m in mol_list]
    # print('mol_to_tokens takes %.3f sec.' % (time() - start_time))
    #
    # # start_time = time()
    # # token_list2 = [smiles_to_token(mol_to_smiles(m)) for m in mol_list]
    # # print('smiles_to_tokens takes %.3f sec.' % (time() - start_time))
    #
    # consistency = [(t1 == t2).all()
    #                for t1, t2 in zip(token_list1, token_list2)]
    # for i, c in enumerate(consistency):
    #     if not c:
    #         print('Chemical %s is tokenized differently in two '
    #               'methods ...' % smiles_list[i])
    #
    # inconsistent_smiles_list = [smiles_list[i]
    #                             for i, c in enumerate(consistency) if not c]
