""" 
    File Name:          MoReL/drug_clustering.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               5/16/19
    Python Version:     3.5.4
    File Description:   

"""
from drug_resp_dataset import *


NUM_BINS = 5
DATA_SOURCES = ['NCI60', ]

PCA_NUM_COMPONENT = 50


try:
    drug_auc_dict = np.load(file='../data/processed/drug_auc_dict.npy',
                            allow_pickle=True).item()
    drug_hist_dict = np.load(file='../data/processed/drug_hist_dict.npy',
                             allow_pickle=True).item()

except FileNotFoundError:

    print('Files not found. Performing pre-processing ... ')

    resp_array = get_resp_array(
        data_path='../data/raw/combined_single_response_agg',
        data_sources=DATA_SOURCES)

    drug_df = load_drug_data(
        data_dir='../data/drug/',
        data_type=DrugDataType.SMILES,
        nan_processing=NanProcessing.NONE)

    drug_auc_dict = {}
    for row in resp_array:

        source, cell_id, drug_id, auc = tuple(row)

        try:
            smiles = drug_df.loc[drug_id].item()
            mol = Chem.MolFromSmiles(smiles)
            assert mol
        except:
            continue

        if drug_id in drug_auc_dict:
            drug_auc_dict[drug_id].append((cell_id, auc))
        else:
            drug_auc_dict[drug_id] = [(cell_id, auc), ]

    # Generate the histogram of AUC for each drug
    bin_interval = 1.0 / NUM_BINS
    drug_hist_dict = {}
    for drug_id, auc_list in drug_auc_dict.items():
        for cell_id, auc in auc_list:

            if drug_id not in drug_hist_dict:
                drug_hist_dict[drug_id] = np.zeros(shape=(NUM_BINS, ))

            bin = int(np.floor(auc / bin_interval))
            # In case of AUC == 1.0
            drug_hist_dict[drug_id][bin if bin < NUM_BINS \
                else (NUM_BINS - 1)] += 1

    # Normalize the histogram to percentage
    for drug_id, auc_hist in drug_hist_dict.items():
        drug_hist_dict[drug_id] = auc_hist / np.sum(auc_hist)

    np.save(file='../data/processed/drug_auc_dict', arr=drug_auc_dict)
    np.save(file='../data/processed/drug_hist_dict', arr=drug_hist_dict)


# #############################################################################
# Distance matrix based on drug AUC curve
# Cluster the drugs based on PCA-ed distance matrix
# #############################################################################
# import cv2
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
#
#
# HISTCMP = cv2.HISTCMP_BHATTACHARYYA
#
#
# try:
#     distance_mat = np.load(file='../data/processed/distance_mat.npy',
#                            allow_pickle=True)
# except FileNotFoundError:
#
#     distance_mat = np.zeros(shape=(len(drug_hist_dict), len(drug_hist_dict)))
#     for row_idx, (_, auc_hist) in enumerate(drug_hist_dict.items()):
#         for col_idx, (_, cmp_auc_hist) in enumerate(drug_hist_dict.items()):
#
#             distance = cv2.compareHist(auc_hist.astype(np.float32),
#                                        cmp_auc_hist.astype(np.float32),
#                                        HISTCMP)
#
#             distance_mat[row_idx, col_idx] = distance
#
#     np.save(file='../data/processed/distance_mat', arr=distance_mat)

# print('Performing PCA with n_component={PCA_NUM_COMPONENT} ... ')
# pca = PCA(n_components=PCA_NUM_COMPONENT)
# drug_pca_distances = pca.fit_transform(distance_mat)
# print(f'PCA expalined variance ratio:\n{pca.explained_variance_ratio_}')
#
#
# print(f'Performing tSNE ...')
# tnse = TSNE(n_components=2)
# drug_tsne_distances = tnse.fit_transform(drug_pca_distances)
#
#
# print(f'Plotting ...')
# plt.figure(figsize=(100, 100), dpi=100)
# plt.scatter(drug_tsne_distances[:, 0], drug_tsne_distances[:, 1])
# for i, (drug_id, auc_hist) in enumerate(drug_hist_dict.items()):
#     auc_list = drug_auc_dict[drug_id]
#     mean_auc = 0.
#     for cell_id, auc in auc_list:
#         mean_auc += auc
#     mean_auc = mean_auc / len(auc_list)
#     annotation = str(drug_id) + f'({mean_auc:0.2f})'
#     plt.annotate(annotation,
#                  (drug_tsne_distances[i, 0], drug_tsne_distances[i, 1]))
# plt.title('Visualized drugs based on AUC')
# plt.savefig(fname=f'drug_tnse({PCA_NUM_COMPONENT}-2)_distances.png')


# #############################################################################
# Label all the drugs based on AUC curves and check the prediction accuracy
# #############################################################################
from rdkit import Chem
from torch.utils.data import Dataset


RDLogger.logger().setLevel(RDLogger.CRITICAL)


drug_label_array = []
for i, (drug_id, auc_hist) in enumerate(drug_hist_dict.items()):

    ordered_indices = auc_hist.argsort()[::-1]
    label = ordered_indices[0] if ordered_indices[0] >= 2 else 2

    drug_label_array.append([drug_id, label])

    if np.abs(ordered_indices[0] - ordered_indices[1]) > 1:
        print(f'Drug {drug_id} has interesting AUC: {list(auc_hist)}.')

drug_label_array = np.array(drug_label_array)
labels, counts = np.unique(drug_label_array[:, 1], return_counts=True)


# Load the drug SMILES strings
drug_smiles_dict = dataframe_to_dict(
        load_drug_data(data_dir='../data/drug/',
                       data_type=DrugDataType.SMILES,
                       nan_processing=NanProcessing.NONE), dtype=str)

drug_graph_dict = featurize_drug_dict(drug_dict=drug_smiles_dict,
                                      featurizer=mol_to_graph,
                                      featurizer_kwargs=None)


class GraphToAUCType(Dataset):

    def __int__(self,
                labels_: np.array,
                drug_label_array_: np.array,
                drug_graph_dict_: dict):

        super().__init__()

        self.__labels = labels_
        self.__drug_label_array = drug_label_array_
        self.__drug_graph_dict = drug_graph_dict_
        # TODO: make sure that all durg ids are in the drug graph dict

        self.__len = len(self.__drug_label_array)

    def __len__(self):
        return self.__len

    def __getitem__(self, index: int):

        __drug_label = self.__drug_label_array[index]
        __drug_id, __label = __drug_label[0], __drug_label[1]

        __graph = self.__drug_graph_dict[__drug_id]
        __target = np.where(labels == __label)[0].item()




