""" 
    File Name:          MoReL/drug_resp_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               4/22/19
    Python Version:     3.5.4
    File Description:   

"""
import os
import torch
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from enum import Enum, auto
from typing import Union, Optional, List

from rdkit import Chem, RDLogger
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, \
    MaxAbsScaler, RobustScaler

from utils.dataset.featurizers import mol_to_tokens, mol_to_graph
# from utils.dataset.featurizers import mol_to_image, mol_to_jtnn

# Suppress unnecessary RDkit warnings and errors
RDLogger.logger().setLevel(RDLogger.CRITICAL)
logger = logging.getLogger(__name__)

# Valid data sources. Used for one-hot-encoding
DATA_SOURCES = ['CCLE', 'CTRP', 'GDSC', 'NCI60', 'gCSI']

# Scaler type union
Scaler = Union[MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler]

# Data loading process for drug response ######################################
# 1. Load drug data (dict), cell data(dict), and response data (array)
# 2. Convert all features into numeric torch tensor.
#    If the drug feature is image, graph, or token (smiles):
#    (a) transform all the smiles into RDkit.Mol object;
#    (b) transform all the Mols into feature
#    And drop all the drugs that are invalid in (a) and (b)
# 3. Extract common drugs and cells, and down-sizing all three data
# 4. Create training and testing Datasets


# Helper functions and ###################################################
def dataframe_to_dict(dataframe: pd.DataFrame,
                      dtype: type = None) -> dict:
    ret = {}
    for index, row in dataframe.iterrows():
        if len(row) == 1:
            value = row.values[0]
            if dtype:
                value = dtype(value)
        else:
            value: np.array = row.values
            if dtype:
                value: np.array = value.astype(dtype=dtype)
        ret[index] = value
    return ret


def scale_dict(data_dict: dict,
               scaler: Optional[Scaler],
               base_keys: np.array or iter = None):

    if scaler is None:
        return data_dict

    __fit_data = [data_dict[k] for k in
                  (data_dict.keys() if (base_keys is None) else base_keys)]

    try:
        scaler.fit(__fit_data)
        __new_values = scaler.transform(list(data_dict.values()))
    except ValueError as e:
        logger.warning(f'Scaling function has encountered ValueError {e}. '
                       f'Using unscaled data.')
        return data_dict

    __ret_dict = {k: np.array(v, dtype=np.float32)
                  for k, v in zip(data_dict.keys(), __new_values)}

    return __ret_dict


def tensorize_dict(data_dict: dict):
    t = type(list(data_dict.values())[0])
    return {k: torch.from_numpy(v) for k, v in data_dict.items()} \
        if (t is np.array) or (t is np.ndarray) else data_dict


# Cell line data ##############################################################
class CellDataType(Enum):
    SNP = 'snp'
    TYPE = 'type'
    RNASEQ = 'rnaseq'


class CellSubsetType(Enum):
    COMPLETE = ''
    LINCS1000 = 'lincs1000'
    ONCOGENES = 'oncogenes'
    MICRO_ARRAY = 'microarray'


class CellProcessingMethod(Enum):
    ORIGINAL = ''
    COMBAT = 'combat'
    AUTOENCODER = 'autoencoder'
    SOURCE_SCALE = 'source_scale'


def load_cell_data(data_dir: str,
                   data_type: CellDataType or str,
                   subset_type: CellSubsetType or str,
                   processing_method: CellProcessingMethod or str,
                   cell_type_subset: Optional[List[str]] or int):

    data_type = CellDataType(data_type)
    subset_type = CellSubsetType(subset_type)
    processing_method = CellProcessingMethod(processing_method)

    # Load the cell line type dataframe if type list is given, which means
    # that we only need a subset of the cell lines of the given types
    if cell_type_subset:
        cell_type_file_path = os.path.join(data_dir, 'combined_type.csv')
        cell_type_df = pd.read_csv(cell_type_file_path, header=0, index_col=0)

        # If cell type is a list, then use it as a subset,
        # Otherwise, if a integer N is given, use it to select the top N
        # cell line types

        if isinstance(cell_type_subset, list):
            cell_type_list = cell_type_subset
        else:
            cell_type_counts = cell_type_df['Type'].value_counts()

            # Make sure that the top N cell line types are within range
            assert cell_type_subset > 0
            cell_type_subset = min(cell_type_subset, len(cell_type_counts))

            cell_type_list = list(
                cell_type_counts[cell_type_counts >=
                                 cell_type_counts[cell_type_subset - 1]].index)

            # If the top N is ambiguous, which means that the (N)th type has
            # the same count as the (N+1)th count, then there will be more
            # than N types in the list
            if len(cell_type_list) > cell_type_subset:
                logger.warning(
                    f'Top {cell_type_subset} cell line types are ambiguous '
                    f'because of tied occurence counts. '
                    f'Using top {len(cell_type_list)} types instead.')

        cell_set = set(
            cell_type_df.loc[cell_type_df['Type'].isin(cell_type_list)].index)
    else:
        cell_set = None

    file_name_list = ['combined', data_type.value]
    if subset_type != CellSubsetType.COMPLETE:
        file_name_list.append(subset_type.value)
    if processing_method != CellProcessingMethod.ORIGINAL:
        file_name_list.append(processing_method.value)

    file_name = '_'.join(file_name_list) + '.csv'
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):

        cell_df = pd.read_csv(file_path, header=0, index_col=0)

        # Down-select cell lines based on given types
        if cell_set:
            cell_df = cell_df[cell_df.index.isin(cell_set)]

        # One-hot encoding if the data_type is CellDataType.TYPE
        if data_type == CellDataType.TYPE:
            cell_df = pd.get_dummies(cell_df, prefix=['Type'])

        return cell_df
    else:
        raise FileExistsError(
            f'{file_path} does not exist. '
            f'Please check the data directory and parameters')


# Drug data ###################################################################
class DrugDataType(Enum):
    SMILES = 'smiles'
    DRAGON7_PFP = 'dragon7_PFP'
    DRAGON7_ECFP = 'dragon7_ECFP'
    DRAGON7_DESCRIPTOR = 'dragon7_descriptors'
    MORDRED_DESCRIPTOR = 'mordred_descriptors'


class DrugFeatureType(Enum):
    # Features that takes SMILES strings and perform transformation
    GRAPH = (DrugDataType.SMILES, mol_to_graph)
    TOKENIZED_SMILES = (DrugDataType.SMILES, mol_to_tokens)
    # TODO: images, JTNN features

    # Features that can be directly loaded
    DRAGON7_PFP = (DrugDataType.DRAGON7_PFP, None)
    DRAGON7_ECFP = (DrugDataType.DRAGON7_ECFP, None)
    DRAGON7_DESCRIPTOR = (DrugDataType.DRAGON7_DESCRIPTOR, None)
    MORDRED_DESCRIPTOR = (DrugDataType.MORDRED_DESCRIPTOR, None)


class NanProcessing(Enum):
    NONE = auto()
    FILL_ZERO = auto()
    DELETE_ROW = auto()
    DELETE_COL = auto()
    FILL_COLUMN_AVERAGE = auto()


def load_drug_data(data_dir: str,
                   data_type: DrugDataType or str,
                   nan_processing: NanProcessing or str):

    data_type = DrugDataType(data_type)
    nan_processing = NanProcessing(nan_processing)

    file_name = '_'.join(['combined', data_type.value]) + '.csv'
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        dataframe = pd.read_csv(file_path, header=0, index_col=0)

        if nan_processing == NanProcessing.FILL_ZERO:
            dataframe.fillna(0., inplace=True)
        elif nan_processing == NanProcessing.DELETE_ROW:
            logger.warning('Deleting rows with NaN values for drug features '
                           'might remove all features!')
            dataframe.dropna(inplace=True)
        elif nan_processing == NanProcessing.DELETE_COL:
            dataframe.dropna(axis='columns', inplace=True)
        elif nan_processing == NanProcessing.FILL_COLUMN_AVERAGE:
            dataframe.dropna(axis='columns', how='all', inplace=True)
            dataframe.fillna(dataframe.mean(), inplace=True)
        else:
            pass

        return dataframe
    else:
        raise FileExistsError(
            f'{file_path} does not exist. '
            f'Please check the data directory and parameters')


def featurize_drug_dict(drug_dict: dict,
                        featurizer: callable,
                        featurizer_kwargs: Optional[dict]):

    if featurizer is None:
        return drug_dict

    __drug_dict = {}
    __featurizer_kwargs = featurizer_kwargs if featurizer_kwargs else {}
    for drug_id, smiles in drug_dict.items():
        try:
            mol: Chem.Mol = Chem.MolFromSmiles(smiles)
            assert mol
        except AssertionError:
            print(f'Failed converting drug with ID {drug_id} '
                  f'from SMILES \'{smiles}\' into Mol object')
            continue

        try:
            feature = featurizer(mol, **__featurizer_kwargs)
            assert (feature is not None)
        except:
            print(f'Successfully converted drug with ID {drug_id} '
                  f'from SMILES \'{smiles}\' into Mol object, '
                  f'but failed to featurize it using {featurizer} '
                  f'with parameters {__featurizer_kwargs}.')
            continue
        __drug_dict[drug_id] = feature
    return __drug_dict


# Drug response data ##########################################################
def get_resp_array(data_path: str,
                   aggregated: bool,
                   target: str = 'AUC',
                   data_sources: Optional[List[str]] = None) -> np.array:

    if aggregated:
        resp_df = pd.read_csv(data_path,
                              sep='\t',
                              header=0,
                              index_col=None,
                              usecols=['SOURCE', 'CELL', 'DRUG', target])
    else:
        if target != 'GROWTH':
            raise ValueError(f'The prediction target of does-dependent drug '
                             f'response must be growth, not {target}.')
        resp_df = pd.read_csv(data_path,
                              sep='\t',
                              header=0,
                              index_col=None,
                              usecols=['SOURCE', 'DRUG_ID', 'CELLNAME',
                                       'LOG_CONCENTRATION', 'GROWTH'])

        # Change the column ordering to align with the aggregated dataframe
        cols = list(resp_df.columns)
        c1 = cols.index('SOURCE')
        c2 = cols.index('DRUG_ID')
        c3 = cols.index('CELLNAME')
        c4 = cols.index('LOG_CONCENTRATION')
        c5 = cols.index('GROWTH')

        cols[c1], cols[c2], cols[c3], cols[c4], cols[c5] = \
            cols[c1], cols[c3], cols[c2], cols[c5], cols[c4]
        resp_df = resp_df[cols]

    # Down select the data sources if given
    if data_sources:
        resp_df = resp_df.loc[resp_df['SOURCE'].isin(data_sources)]

    resp_array = resp_df.values

    # Change the dtype of prediction target and concentration if exists
    resp_array[:, 3] = np.array(resp_array[:, 3], dtype=np.float32)
    if not aggregated:
        resp_array[:, 4] = np.array(resp_array[:, 4], dtype=np.float32)

    nan_indices = np.isnan(np.float32(resp_array[:, 3]))
    if any(nan_indices):
        logger.warning(
            f'The following lines from \'{data_path}\' contains NaN in the '
            f'\'{target}\' column:\n\t{resp_array[nan_indices]}')

    # Get rid of the NaN values in drug response
    resp_array = resp_array[~nan_indices]

    return resp_array


def trim_resp_array(resp_array: np.array,
                    cells: iter,
                    drugs: iter,
                    inclusive: bool = True) -> np.array:

    cell_set = set(cells)
    drug_set = set(drugs)
    resp_list = []

    for row in resp_array:
        cell_id, drug_id = row[1], row[2]

        if inclusive:
            if (cell_id in cell_set) and (drug_id in drug_set):
                resp_list.append(row)
        else:
            if (cell_id not in cell_set) and (drug_id not in drug_set):
                resp_list.append(row)

    return np.array(resp_list)


# Datasets ####################################################################
def trn_tst_split(resp_array: np.array,
                  rand_state: int = 0,
                  test_ratio: float = 0.2,
                  auc_threshold: float = 0.5,
                  disjoint_cells: bool = True,
                  disjoint_drugs: bool = False):

    # Corner case when test_ratio = 0. or 1., which means that no splitting
    if (test_ratio == 0.) or (test_ratio == 1.):
        __empty_resp_array = np.empty((0, resp_array.shape[1]))
        return (resp_array, __empty_resp_array) if (test_ratio == 0.) \
            else (__empty_resp_array, resp_array)

    # If drugs and cells are not specified to be disjoint in the training
    # and testing dataset, then random split stratified on data sources and AUC
    if (not disjoint_cells) and (not disjoint_drugs):

        __source_array = resp_array[:, 0].reshape(-1, 1)
        __auc_bins = (resp_array[:, 3] > auc_threshold).reshape(-1, 1)
        __stratify = np.concatenate((__source_array, __auc_bins),  axis=1)

        return train_test_split(resp_array,
                                test_size=test_ratio,
                                random_state=rand_state,
                                stratify=__stratify)

    # Note: make sure that the cell and drug column are 1 and 2 separately
    __cell_array = np.unique(resp_array[:, 1])
    __drug_array = np.unique(resp_array[:, 2])

    # Adjust the split ratio if both cells and drugs are disjoint
    # Note that mathematically speaking, we should adjust
    __test_ratio = test_ratio ** 0.7 if (disjoint_cells and disjoint_drugs) \
        else test_ratio

    if disjoint_cells:
        __trn_cell_array, __tst_cell_array = \
            train_test_split(__cell_array,
                             test_size=__test_ratio,
                             random_state=rand_state)
    else:
        __trn_cell_array, __tst_cell_array = __cell_array, __cell_array

    if disjoint_drugs:
        __trn_drug_array, __tst_drug_array = \
            train_test_split(__drug_array,
                             test_size=__test_ratio,
                             random_state=rand_state)
    else:
        __trn_drug_array, __tst_drug_array = __drug_array, __drug_array

    __trn_resp_array = trim_resp_array(resp_array,
                                       cells=__trn_cell_array,
                                       drugs=__trn_drug_array)
    __tst_resp_array = trim_resp_array(resp_array,
                                       cells=__tst_cell_array,
                                       drugs=__tst_drug_array)
    return __trn_resp_array, __tst_resp_array


class ScalingMethod(Enum):
    NONE = None
    MINMAX_1 = MinMaxScaler(feature_range=(0, 1))
    MINMAX_2 = MinMaxScaler(feature_range=(-1, 1))
    STANDARD = StandardScaler()


def scale_feature(trn_resp_array: np.array,
                  cell_dict: dict,
                  drug_dict: dict,
                  cell_scaler: ScalingMethod or Scaler,
                  drug_scaler: ScalingMethod or Scaler):

    if type(cell_scaler) is ScalingMethod:
        # Using deep copy here will make sure that different feature will
        # use a fresh scaler every time this function is called. Although
        # the results are the same because of the sklearn implementation,
        # but still it is conceptually correct to use deepcopy here.
        cell_scaler: Optional[Scaler] = deepcopy(cell_scaler.value)

    new_cell_dict = cell_dict if cell_scaler is None \
        else scale_dict(data_dict=cell_dict,
                        scaler=cell_scaler,
                        base_keys=np.unique(trn_resp_array[:, 1]))

    if type(drug_scaler) is ScalingMethod:
        drug_scaler: Optional[Scaler] = drug_scaler.value

    new_drug_dict = drug_dict if drug_scaler is None \
        else scale_dict(data_dict=drug_dict,
                        scaler=drug_scaler,
                        base_keys=np.unique(trn_resp_array[:, 2]))

    return new_cell_dict, new_drug_dict


class SubsampleType(Enum):
    ON_RECORD = 'record'
    ON_CELL = 'cell'
    ON_DRUG = 'drug'


class DrugRespDataset(Dataset):

    def __init__(self,
                 cell_dict: dict,
                 drug_dict: dict,
                 resp_array: np.array,
                 aggregated: bool):

        super().__init__()

        self.__cell_dict = cell_dict
        self.__drug_dict = drug_dict
        self.__resp_array = resp_array

        self.__sources = DATA_SOURCES.copy()

        self.__aggregated = (self.__resp_array.shape[1] == 4)
        assert (self.__aggregated == aggregated)

        self.__dose_info = 'dose-independent' if self.__aggregated \
            else 'dose-dependent'
        self.__info = self.__get_info()
        self.__len = len(self.__resp_array)

    def __get_info(self):
        return \
            f'This {self.__dose_info} drug response dataset contains:\n'\
            f'\t{len(self.__resp_array)} response records from '\
            f'{len(np.unique(self.__resp_array[:,0]))} different sources;\n'\
            f'\t{len(np.unique(self.__resp_array[:,1]))} unique cell lines;\n'\
            f'\t{len(np.unique(self.__resp_array[:,2]))} unique drugs.'

    def __str__(self):
        if self.__info is None:
            self.__info = self.__get_info()
        return self.__info

    def __len__(self):
        return self.__len

    def __getitem__(self, index: int):

        resp_data = self.__resp_array[index]
        source, cell_id, drug_id, target = \
            resp_data[0], resp_data[1], resp_data[2], resp_data[3]

        source_data = np.zeros_like(self.__sources, dtype=np.float32)
        source_data[self.__sources.index(source)] = 1.
        source_data = torch.from_numpy(source_data)

        cell_data = self.__cell_dict[cell_id]

        drug_data = self.__drug_dict[drug_id]

        target_data = torch.from_numpy(np.array([target, ], dtype=np.float32))

        if not self.__aggregated:
            concentration = resp_data[4]
            concentration_data = torch.from_numpy(
                np.array([concentration, ], dtype=np.float32))
        else:
            concentration_data = None

        return source_data, cell_data, drug_data, \
            target_data, concentration_data

    def subsample(self,
                  subsample_type: SubsampleType,
                  remaining_percentage: float):

        assert (remaining_percentage > 0.) and (remaining_percentage <= 1.)

        if subsample_type == SubsampleType.ON_RECORD:
            remaining_num_records = \
                int(remaining_percentage * len(self.__resp_array))
            self.__resp_array = \
                self.__resp_array[np.random.choice(
                    self.__resp_array.shape[0],
                    remaining_num_records,
                    replace=False), :]

        elif subsample_type == SubsampleType.ON_CELL:
            cells = np.unique(self.__resp_array[:, 1])
            remaining_num_cells = \
                int(remaining_percentage * len(cells))
            remaining_cells = cells[np.random.choice(
                len(cells), remaining_num_cells, replace=False)]

            self.__resp_array = self.__resp_array[
                np.in1d(self.__resp_array[:, 1], remaining_cells)]

        elif subsample_type == SubsampleType.ON_DRUG:
            drugs = np.unique(self.__resp_array[:, 2])
            remaining_num_drugs = \
                int(remaining_percentage * len(drugs))
            remaining_drugs = drugs[np.random.choice(
                len(drugs), remaining_num_drugs, replace=False)]

            self.__resp_array = self.__resp_array[
                np.in1d(self.__resp_array[:, 2], remaining_drugs)]

        else:
            return

        # Change the length and info
        self.__info = self.__get_info()
        self.__len = len(self.__resp_array)


def get_datasets(
        resp_data_path: str,
        resp_aggregated: bool,
        resp_target: str,
        resp_data_sources: Optional[List[str]],

        cell_data_dir: str,
        cell_data_type: CellDataType or str,
        cell_subset_type: CellSubsetType or str,
        cell_processing_method: CellProcessingMethod or str,
        cell_scaling_method: ScalingMethod or Scaler,
        cell_type_subset: Optional[List[str]] or int,

        drug_data_dir: str,
        drug_feature_type: DrugFeatureType or tuple,
        drug_nan_processing: NanProcessing or str,
        drug_scaling_method: ScalingMethod or Scaler,
        drug_featurizer_kwargs: dict = None,

        rand_state: int = 0,
        test_ratio: float = 0.2,
        disjoint_cells: bool = True,
        disjoint_drugs: bool = False,

        summary: bool = False):

    # TODO: multi-feature of arbitrary combination

    # 1. Load drug data (dict), cell data(dict), and response data (array)
    # 2. Convert all the data to numeric torch tensor
    cell_dict = dataframe_to_dict(
        load_cell_data(data_dir=cell_data_dir,
                       data_type=cell_data_type,
                       subset_type=cell_subset_type,
                       processing_method=cell_processing_method,
                       cell_type_subset=cell_type_subset),
        dtype=np.float32)

    drug_data_type: DrugDataType = drug_feature_type.value[0]
    drug_featurizer: callable = drug_feature_type.value[1]

    if drug_featurizer and (drug_data_type != DrugDataType.SMILES):
        raise ValueError(f'Featurizer {drug_featurizer} requires loading '
                         f'SMILES strings, not {drug_data_type}.')

    tmp_drug_dict = dataframe_to_dict(
        load_drug_data(data_dir=drug_data_dir,
                       data_type=drug_data_type,
                       nan_processing=drug_nan_processing),
        dtype=(str if drug_featurizer else np.float32))

    drug_dict = featurize_drug_dict(drug_dict=tmp_drug_dict,
                                    featurizer=drug_featurizer,
                                    featurizer_kwargs=drug_featurizer_kwargs)

    resp_array = get_resp_array(data_path=resp_data_path,
                                aggregated=resp_aggregated,
                                target=resp_target,
                                data_sources=resp_data_sources)

    # 3. Extract common drugs and cells, and down-sizing all three data
    resp_array = trim_resp_array(resp_array=resp_array,
                                 cells=cell_dict.keys(),
                                 drugs=drug_dict.keys(),
                                 inclusive=True)

    trn_resp_array, tst_resp_array = \
        trn_tst_split(resp_array=resp_array,
                      rand_state=rand_state,
                      test_ratio=test_ratio,
                      disjoint_cells=disjoint_cells,
                      disjoint_drugs=disjoint_drugs)

    # 4. Feature scaling for drugs and cells
    if (drug_featurizer is not None) and \
        ((drug_scaling_method is not ScalingMethod.NONE) or
         (drug_scaling_method is not None)):
        logger.warning(f'Cannot perform scaling on drug of featurizer '
                       f'function {drug_featurizer}. '
                       f'Changing the scaling to None ...')
        drug_scaling_method = ScalingMethod.NONE

    cell_dict, drug_dict = \
        scale_feature(trn_resp_array=trn_resp_array,
                      cell_dict=cell_dict,
                      drug_dict=drug_dict,
                      cell_scaler=cell_scaling_method,
                      drug_scaler=drug_scaling_method)

    # 5. Create training and testing Datasets
    cell_dict = tensorize_dict(cell_dict)
    drug_dict = tensorize_dict(drug_dict)

    trn_dataset = DrugRespDataset(cell_dict=cell_dict,
                                  drug_dict=drug_dict,
                                  resp_array=trn_resp_array,
                                  aggregated=resp_aggregated)

    tst_dataset = DrugRespDataset(cell_dict=cell_dict,
                                  drug_dict=drug_dict,
                                  resp_array=tst_resp_array,
                                  aggregated=resp_aggregated)

    if summary:
        print(f'Training set length {len(trn_dataset)}; '
              f'testing set length {len(tst_dataset)}.')

    return trn_dataset, tst_dataset, cell_dict, drug_dict


# Testing segment
if __name__ == '__main__':

    # trn_dset, tst_dset = get_datasets(
    #     resp_data_path='../../data/raw/combined_single_response_agg',
    #     resp_target='AUC',
    #
    #     cell_data_dir='../../data/cell/',
    #     cell_data_type=CellDataType.RNASEQ,
    #     cell_subset_type=CellSubsetType.LINCS1000,
    #     cell_processing_method=CellProcessingMethod.SOURCE_SCALE,
    #     cell_scaling_method=ScalingMethod.NONE,
    #
    #     drug_data_dir='../../data/drug/',
    #     drug_feature_type=DrugFeatureType.TOKENIZED_SMILES,
    #     drug_nan_processing=NanProcessing.FILL_COLUMN_AVERAGE,
    #     drug_scaling_method=ScalingMethod.STANDARD,
    #     drug_featurizer_kwargs={'len_tokens': 256})

    # trn_dset, tst_dset = get_datasets(
    #     resp_data_path='../../data/raw/combined_single_response_agg',
    #     resp_target='AUC',
    #
    #     cell_data_dir='../../data/cell/',
    #     cell_data_type=CellDataType.RNASEQ,
    #     cell_subset_type=CellSubsetType.LINCS1000,
    #     cell_processing_method=CellProcessingMethod.SOURCE_SCALE,
    #     cell_scaling_method=ScalingMethod.NONE,
    #
    #     drug_data_dir='../../data/drug/',
    #     drug_feature_type=DrugFeatureType.GRAPH,
    #     drug_nan_processing=NanProcessing.NONE,
    #     drug_scaling_method=ScalingMethod.STANDARD)

    import time
    start_time = time.time()

    trn_dset, _, trn_cell_dict, trn_drug_dict = get_datasets(
        resp_data_path='/raid/xduan7/Data/combined_single_drug_growth.txt',
        resp_aggregated=False,
        resp_target='GROWTH',
        resp_data_sources=['CTRP', ],

        cell_data_dir='/raid/xduan7/Data/cell/',
        cell_data_type=CellDataType.RNASEQ,
        cell_subset_type=CellSubsetType.LINCS1000,
        cell_processing_method=CellProcessingMethod.SOURCE_SCALE,
        cell_scaling_method=ScalingMethod.NONE,
        cell_type_subset=None,

        drug_data_dir='/raid/xduan7/Data/drug/',
        drug_feature_type=DrugFeatureType.DRAGON7_DESCRIPTOR,
        drug_nan_processing=NanProcessing.DELETE_COL,
        drug_scaling_method=ScalingMethod.STANDARD,

        rand_state=0,
        test_ratio=0.,
        disjoint_drugs=False,
        disjoint_cells=False)

    tmp_resp_array = get_resp_array(
        data_path='/raid/xduan7/Data/combined_single_drug_growth.txt',
        aggregated=False,
        target='GROWTH',
        data_sources=['GDSC', ])
    tmp_resp_array = trim_resp_array(
        resp_array=tmp_resp_array,
        cells=trn_cell_dict.keys(),
        drugs=trn_drug_dict.keys(),
        inclusive=True)
    tst_dset = DrugRespDataset(
        cell_dict=trn_cell_dict,
        drug_dict=trn_drug_dict,
        resp_array=tmp_resp_array,
        aggregated=False)

    print('%' * 8 + ' Training Set Info:\n' + str(trn_dset))
    print('%' * 8 + ' Testing Set Info:\n' + str(tst_dset))

    print(f'Created datasets in {time.time() - start_time: .2f} seconds.')
