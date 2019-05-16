""" 
    File Name:          MoReL/drug_clustering.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               5/16/19
    Python Version:     3.5.4
    File Description:   

"""
import cv2
from rdkit import Chem
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from drug_resp_dataset import *


RDLogger.logger().setLevel(RDLogger.CRITICAL)


NUM_BINS = 4
DATA_SOURCES = ['NCI60', ]
HISTCMP = cv2.HISTCMP_BHATTACHARYYA
PCA_NUM_COMPONENT = 50


try:
    drug_auc_dict = np.load(file='drug_auc_dict.npy',
                            allow_pickle=True).item()
    drug_hist_dict = np.load(file='drug_hist_dict.npy',
                             allow_pickle=True).item()
    distance_mat = np.load(file='distance_mat.npy', allow_pickle=True)

except FileNotFoundError:

    print('Files not found. Performing preprocessing ... ')

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

    distance_mat = np.zeros(shape=(len(drug_hist_dict), len(drug_hist_dict)))
    for row_idx, (_, auc_hist) in enumerate(drug_hist_dict.items()):
        for col_idx, (_, cmp_auc_hist) in enumerate(drug_hist_dict.items()):

            distance = cv2.compareHist(auc_hist.astype(np.float32),
                                       cmp_auc_hist.astype(np.float32),
                                       HISTCMP)

            distance_mat[row_idx, col_idx] = distance

    np.save(file='drug_auc_dict', arr=drug_auc_dict)
    np.save(file='drug_hist_dict', arr=drug_hist_dict)
    np.save(file='distance_mat', arr=distance_mat)


print('Performing PCA with n_component={PCA_NUM_COMPONENT} ... ')
pca = PCA(n_components=PCA_NUM_COMPONENT)
drug_pca_distances = pca.fit_transform(distance_mat)
print(f'PCA expalined variance ratio:\n{pca.explained_variance_ratio_}')


print(f'Performing tSNE ...')
tnse = TSNE(n_components=2)
drug_tsne_distances = tnse.fit_transform(drug_pca_distances)


print(f'Plotting ...')
plt.figure(figsize=(100, 100), dpi=100)
plt.scatter(drug_tsne_distances[:, 0], drug_tsne_distances[:, 1])
for i, (drug_id, auc_hist) in enumerate(drug_hist_dict.items()):
    auc_list = drug_auc_dict[drug_id]
    mean_auc = 0.
    for cell_id, auc in auc_list:
        mean_auc += auc
    mean_auc = mean_auc / len(auc_list)
    annotation = str(drug_id) + f'({mean_auc:0.2f})'
    plt.annotate(annotation,
                 (drug_tsne_distances[i, 0], drug_tsne_distances[i, 1]))
plt.title('Visualized drugs based on AUC')
plt.savefig(fname=f'drug_tnse({PCA_NUM_COMPONENT}-2)_distances.png')
