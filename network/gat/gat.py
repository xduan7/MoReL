""" 
    File Name:          MoReL/gat.py
    Author:             Xiaotian Duan (xduan)
    Email:              xduan7@uchicago.edu
    Date:               4/16/2019
    Python Version:     3.5.4
    File Description:   

"""
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data


class GAT(nn.Module):

    def __init__(self,
                 node_attr_dim: int,
                 state_dim: int = 8,
                 num_heads: int = 8,
                 num_conv: int = 2,
                 out_dim: int = 1,
                 dropout_rate: float = 0.2):

        super(GAT, self).__init__()

        __conv_layers = nn.ModuleList(
            [pyg_nn.GATConv(node_attr_dim, state_dim,
                            heads=num_heads, dropout=dropout_rate)])

        for i in range(num_conv - 1):
            __conv_layers.extend(
                [nn.ReLU(),
                 nn.Dropout(dropout_rate),
                 pyg_nn.GATConv(state_dim,
                                out_dim if (i == num_conv - 2) else state_dim,
                                heads=num_heads, dropout=dropout_rate)])

    def forward(self, data: pyg_data.Data):
        return self.__conv_layers(data.x, data.edge_index)


class EdgeGAT(nn.Module):
    """
    Version of GCN that takes one-hot encoded edge attribute
    """

    def __init__(self,
                 node_attr_dim: int,
                 edge_attr_dim: int,
                 state_dim: int = 8,
                 num_heads: int = 8,
                 num_conv: int = 2,
                 out_dim: int = 1,
                 dropout_rate: float = 0.2):

        super(EdgeGAT, self).__init__()

        self.__edge_attr_dim = edge_attr_dim
        __gat_kwargs = {
            'node_attr_dim': node_attr_dim,
            'state_dim': state_dim,
            'num_heads': num_heads,
            'num_conv': num_conv,
            'out_dim': out_dim,
            'dropout_rate': dropout_rate}

        self.__gat_nets = nn.ModuleList(
            [GAT(**__gat_kwargs) for _ in range(edge_attr_dim)])

    def forward(self, data: pyg_data.Data):

        out = []
        for i in range(self.__edge_attr_dim):
            # New graph that corresponds to the edge attributes
            _edge_index = torch.masked_select(
                data.edge_index, mask=data.edge_attr[:, i].byte()).view(2, -1)
            _data = pyg_data.Data(x=data.x, edge_index=_edge_index)
            out.append(self.__gat_nets[i](_data))

        return torch.cat(tuple(out), dim=1)


# Testing segment for GCN with edge attributes and pooling layer
if __name__ == '__main__':

    import torch
    import numpy as np
    import pandas as pd
    import utils.data_prep.config as c
    from utils.dataset.graph_to_dscrptr_dataset import GraphToDscrptrDataset

    PCBA_ONLY = True
    USE_CUDA = True
    RAND_STATE = 0
    TARGET_LIST = c.TARGET_D7_DSCRPTR_NAMES

    use_cuda = torch.cuda.is_available() and USE_CUDA
    device = torch.device('cuda: 0' if use_cuda else 'cpu')

    cid_smiles_csv_path = c.PCBA_CID_SMILES_CSV_PATH \
        if PCBA_ONLY else c.PC_CID_SMILES_CSV_PATH
    cid_smiles_df = pd.read_csv(cid_smiles_csv_path,
                                sep='\t',
                                header=0,
                                index_col=0,
                                dtype=str)
    cid_smiles_df.index = cid_smiles_df.index.map(str)
    cid_smiles_dict = cid_smiles_df.to_dict()['SMILES']
    del cid_smiles_df

    cid_list = []
    dscrptr_array = np.array([], dtype=np.float32).reshape(0, len(TARGET_LIST))
    for chunk_cid_dscrptr_df in pd.read_csv(
            c.PCBA_CID_TARGET_D7DSCPTR_CSV_PATH,
            sep='\t',
            header=0,
            index_col=0,
            usecols=['CID'] + TARGET_LIST,
            dtype={**{'CID': str}, **{t: np.float32 for t in TARGET_LIST}},
            chunksize=2 ** 16):
        chunk_cid_dscrptr_df.index = chunk_cid_dscrptr_df.index.map(str)
        cid_list.extend(list(chunk_cid_dscrptr_df.index))
        dscrptr_array = np.vstack((dscrptr_array, chunk_cid_dscrptr_df.values))

    # Perform STD normalization for multi-target regression
    dscrptr_mean = np.mean(dscrptr_array, axis=0)
    dscrptr_std = np.std(dscrptr_array, axis=0)
    dscrptr_array = (dscrptr_array - dscrptr_mean) / dscrptr_std

    assert len(cid_list) == len(dscrptr_array)
    cid_dscrptr_dict = {cid: dscrptr
                        for cid, dscrptr in zip(cid_list, dscrptr_array)}

    smiles_cid_set = set(list(cid_smiles_dict.keys()))
    dscrptr_cid_set = set(list(cid_dscrptr_dict.keys()))
    cid_list = sorted(list(smiles_cid_set & dscrptr_cid_set), key=int)

    ###########################################################################
    # Dataset and dataloader
    dataset_kwargs = {
        'target_list': TARGET_LIST,
        'cid_smiles_dict': cid_smiles_dict,
        'cid_dscrptr_dict': cid_dscrptr_dict}
    dataset = GraphToDscrptrDataset(cid_list=cid_list, **dataset_kwargs)

    dataloader_kwargs = {
        'batch_size': 32,
        'timeout': 1,
        'pin_memory': True if use_cuda else False,
        'num_workers': 2 if use_cuda else 0}
    dataloader = pyg_data.DataLoader(dataset,
                                     shuffle=True,
                                     **dataloader_kwargs)

    model = EdgeGCNEncoder(node_attr_dim=dataset.node_attr_dim,
                           edge_attr_dim=dataset.edge_attr_dim,
                           out_dim=len(TARGET_LIST)).to(device)

    model.train()
    data = next(iter(dataloader))

    print(f'The input batch data is {data}')
    data = data.to(device)
    print(f'The output shape is {model(data).shape}')

