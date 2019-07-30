""" 
    File Name:          MoReL/graph_drug_response.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               7/28/19
    Python Version:     3.5.4
    File Description:   

"""
from comet_ml import Optimizer
import torch_geometric.data as pyg_data
import torch.nn.functional as F
from sklearn import metrics

# Constant to modify
PROJ_LOCATION = '/vol/ml/xduan7/Projects/MoReL'
DATA_LOCATION = '/vol/ml/xduan7/Data'

import sys
sys.path.extend([PROJ_LOCATION])
from utils.dataset.drug_resp_dataset import *
from network.gnn.gat.gat import EdgeGATEncoder
from network.gnn.gcn.gcn import EdgeGCNEncoder
from network.gnn.mpnn.mpnn import MPNN
from network.simple_uno import SimpleUno

comet_opt = Optimizer(project_name='Drug Response with Graph Models')

# Construct the datasets for training and testing
bigrun_cell_id_list = pd.read_csv(
    DATA_LOCATION + '/bigrun_cell_ids.csv',
    index_col=None).values.reshape((-1)).tolist()
bigrun_drug_id_list = pd.read_csv(
    DATA_LOCATION + '/bigrun_drug_ids.csv',
    index_col=None).values.reshape((-1)).tolist()

trn_dset, tst_dset, _, _ = get_datasets(
    resp_data_path=(DATA_LOCATION +
                    '/combined_single_drug_response_aggregated.csv'),
    resp_aggregated=True,
    resp_target='AUC',
    resp_data_sources=DATA_SOURCES,

    cell_data_dir=(DATA_LOCATION + '/cell/'),
    cell_id_list=bigrun_cell_id_list,
    cell_data_type=CellDataType.RNASEQ,
    cell_subset_type=CellSubsetType.LINCS1000,
    cell_processing_method=CellProcessingMethod.SOURCE_SCALE,
    cell_scaling_method=ScalingMethod.NONE,
    cell_type_subset=None,

    drug_data_dir=(DATA_LOCATION + '/drug/'),
    drug_id_list=bigrun_drug_id_list,
    drug_feature_type=DrugFeatureType.GRAPH,
    drug_nan_processing=NanProcessing.NONE,
    drug_scaling_method=ScalingMethod.NONE,
    drug_featurizer_kwargs=None,

    # Random split
    disjoint_cells=False,
    disjoint_drugs=False,
    summary=True)

node_attr_dim = trn_dset[0].x.shape[1]
edge_attr_dim = trn_dset[0].edge_attr.shape[1]
cell_input_dim = trn_dset[0].cell_data.shape[0]

# Iterate through all different experiment configurations
for experiment in comet_opt.get_experiments():

    # Disable auto-collection of any metrics
    experiment.disable_mp()

    graph_model = experiment.get_parameter(name='graph_model')
    graph_state_dim = experiment.get_parameter(name='graph_state_dim')
    graph_num_conv = experiment.get_parameter(name='graph_num_conv')
    graph_out_dim = experiment.get_parameter(name='graph_out_dim')
    graph_attention_pooling = \
        (experiment.get_parameter(name='graph_attention_pooling') == 'True')
    uno_dropout = experiment.get_parameter(name='uno_dropout')

    uno_state_dim = experiment.get_parameter(name='uno_state_dim')
    cell_state_dim = experiment.get_parameter(name='cell_state_dim')

    bin_auc_num = experiment.get_parameter(name='bin_auc_num')

    batch_size = experiment.get_parameter(name='batch_size')
    num_workers = experiment.get_parameter(name='num_workers')
    max_num_epochs = experiment.get_parameter(name='max_num_epochs')
    learning_rate = experiment.get_parameter(name='learning_rate')
    num_logs_per_epoch = experiment.get_parameter(name='num_logs_per_epoch')
    early_step_patience = experiment.get_parameter(name='early_step_patience')

    # Construct a tag for the uno with graph model
    # pooling_tag = 'attention_pooling' if graph_attention_pooling \
    #     else 'set2set_pooling'
    # graph_model_tag = f'{graph_model}({graph_num_conv}*{graph_state_dim}->' \
    #                   f'{graph_out_dim}+{pooling_tag})'
    # experiment.add_tag(f'[{graph_model_tag}: {cell_state_dim}]->'
    #                    f'[{uno_state_dim}(dropout={uno_dropout})]')

    experiment.add_tags(tags=[
        f'graph_model={graph_model}',
        f'graph_attention_pooling={graph_attention_pooling}', ])

    # Everything in the experiment will be put inside this try statement
    # If exception happens, clean things up and move to the next experiment
    try:
        # Dataloaders
        dataloader_kwargs = {
            'pin_memory': True,
            'batch_size': batch_size,
            'num_workers': num_workers, }
        trn_loader = pyg_data.DataLoader(trn_dset, shuffle=True,
                                         **dataloader_kwargs)
        tst_loader = pyg_data.DataLoader(tst_dset, **dataloader_kwargs)

        # Construct graph model, might run into CUDA memory error
        graph_model_kwargs = {
            'node_attr_dim': node_attr_dim,
            'edge_attr_dim': edge_attr_dim,
            'state_dim': graph_state_dim,
            'num_conv': graph_num_conv,
            'out_dim': graph_out_dim,
            'attention_pooling': graph_attention_pooling, }

        if graph_model == 'gcn':
            drug_tower = EdgeGCNEncoder(**graph_model_kwargs)
        elif graph_model == 'gat':
            drug_tower = EdgeGATEncoder(**graph_model_kwargs)
        else:
            drug_tower = MPNN(**graph_model_kwargs)

        model = SimpleUno(
            state_dim=uno_state_dim,
            dose_info=False,
            cell_state_dim=cell_state_dim,
            drug_state_dim=graph_out_dim,
            cell_input_dim=cell_input_dim,
            drug_tower=drug_tower,
            dropout_rate=uno_dropout,
            sigmoid_output=True).to('cuda')
        experiment.set_model_graph(str(model))

        # Construct optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10, min_lr=(learning_rate/100.))

        # Iterate through epochs
        best_r2 = float('-inf')
        early_stop_counter = 0
        num_batches_per_log = \
            int(np.ceil(len(trn_loader) / num_logs_per_epoch))

        for epoch in range(max_num_epochs):
            with experiment.train():
                model.train()
                __trn_loss, __batch_counter, __sample_counter = 0., 0, 0

                for batch_data in trn_loader:

                    __batch_size = batch_data.num_graphs
                    batch_data = batch_data.to('cuda')
                    cell_data = batch_data.cell_data.view(__batch_size, -1)
                    trgt = batch_data.target_data.view(__batch_size, -1)

                    pred = model(cell_data=cell_data, drug_data=batch_data)
                    loss = F.mse_loss(pred, trgt)
                    loss.backward()
                    optimizer.step()

                    __trn_loss += loss.item() * __batch_size
                    __batch_counter += 1
                    __sample_counter += __batch_size

                    if ((__batch_counter % num_batches_per_log) == 0) or \
                       (__batch_counter == len(trn_loader)):

                        __avg_trn_loss = (__trn_loss / __sample_counter)
                        __step = epoch + (__batch_counter / len(trn_loader))
                        # __step = int(__step * num_logs_per_epoch)

                        experiment.log_metric(name='loss',
                                              value=__avg_trn_loss,
                                              step=__step)
                        __trn_loss, __sample_counter = 0., 0

            with experiment.test():
                model.eval()
                trgt_array = np.zeros(shape=(1, ))
                pred_array = np.zeros(shape=(1, ))

                with torch.no_grad():
                    for batch_data in tst_loader:

                        __batch_size = batch_data.num_graphs
                        batch_data = batch_data.to('cuda')
                        cell_data = batch_data.cell_data.view(__batch_size, -1)
                        trgt = batch_data.target_data.view(__batch_size, -1)

                        pred = model(cell_data=cell_data,
                                     drug_data=batch_data)

                        trgt_array = np.concatenate(
                            (trgt_array, trgt.cpu().numpy().reshape(-1)))
                        pred_array = np.concatenate(
                            (pred_array, pred.cpu().numpy().reshape(-1)))

                    # Compute and log all the metrics
                    # Regression metrics
                    reg_kwargs = {'y_true': trgt_array,
                                  'y_pred': pred_array}

                    nan_indices = np.isnan(trgt_array)
                    if any(nan_indices):
                        print(f'Target array contains NaN: '
                              f'{trgt_array[nan_indices]}:'
                              f'{pred_array[nan_indices]}')
                    nan_indices = np.isnan(pred_array)
                    if any(nan_indices):
                        print(f'Predicted array contains NaN: '
                              f'{trgt_array[nan_indices]}:'
                              f'{pred_array[nan_indices]}')

                    tst_r2 = metrics.r2_score(**reg_kwargs)
                    tst_mae = metrics.mean_absolute_error(**reg_kwargs)
                    tst_mse = metrics.mean_squared_error(**reg_kwargs)

                    # Binary classification metrics
                    bin_trgt_array = \
                        (~np.digitize(trgt_array, [bin_auc_num]).
                         astype(np.bool)).astype(np.int)
                    bin_pred_array = \
                        (~np.digitize(pred_array, [bin_auc_num]).
                         astype(np.bool)).astype(np.int)

                    bin_kwargs = {'y_true': bin_trgt_array,
                                  'y_pred': bin_pred_array}

                    tst_acc = metrics.accuracy_score(**bin_kwargs)
                    tst_bal_acc = metrics.balanced_accuracy_score(**bin_kwargs)
                    tst_mcc = metrics.matthews_corrcoef(**bin_kwargs)
                    # tst_auc = metrics.roc_auc_score(**bin_kwargs)

                    tn, fp, fn, tp = \
                        metrics.confusion_matrix(**bin_kwargs).ravel()

                    tpr = 1. if (tp + fn) == 0 else (tp / (tp + fn))
                    tnr = 1. if (tn + fp) == 0 else (tn / (tn + fp))
                    fpr, fnr = (1 - tnr), (1 - tpr)

                    # Comet log metrics
                    experiment.log_metric('r2', tst_r2, step=(epoch + 1))
                    experiment.log_metric('mae', tst_mae, step=(epoch + 1))
                    experiment.log_metric('mse', tst_mse, step=(epoch + 1))

                    experiment.log_metric('acc', tst_acc, step=(epoch + 1))
                    experiment.log_metric('bal_acc', tst_bal_acc,
                                          step=(epoch + 1))
                    experiment.log_metric('mcc', tst_mcc, step=(epoch + 1))
                    # experiment.log_metric('auc', tst_auc)

                    experiment.log_metric('tpr', tpr, step=(epoch + 1))
                    experiment.log_metric('tnr', tnr, step=(epoch + 1))
                    experiment.log_metric('fpr', fpr, step=(epoch + 1))
                    experiment.log_metric('fnr', fnr, step=(epoch + 1))

            experiment.log_epoch_end(epoch)

            scheduler.step(tst_mse)
            if tst_r2 > best_r2:
                best_r2 = tst_r2
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_step_patience:
                    break

        experiment.log_metric('best_r2', best_r2, include_context=False)

    except Exception as e:
        print('Cannot finish the current experiment.')
        print(e)

        experiment.log_metric('best_r2', 0., include_context=False)

        try:
            del model, optimizer, scheduler, trn_loader, tst_loader
        except NameError:
            pass

        torch.cuda.empty_cache()
        experiment.end()

        continue

