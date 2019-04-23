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
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data


class GAT(nn.Module):

    def __init__(self,
                 node_attr_dim: int,
                 state_dim: int = 8,
                 num_heads: int = 8,
                 num_conv: int = 2,
                 out_dim: int = 1,
                 dropout: float = 0.2):

        super(GAT, self).__init__()
        self.__dropout = dropout

        # Convolution layers
        # Note that there is a difference between dropout in GAT layers and
        # the dropout in-between. The former indicates the dropout of graph
        # model propagation; and the latter is between layers.
        self.__conv_layers = nn.ModuleList([pyg_nn.GATConv(
            node_attr_dim if (i == 0) else state_dim * num_heads,
            out_dim if (i == (num_conv - 1)) else state_dim,
            heads=(1 if (i == (num_conv - 1)) else num_heads),
            dropout=dropout) for i in range(num_conv)])

    def forward(self, data: pyg_data.Data):
        out = data.x
        for i, layer in enumerate(self.__conv_layers):
            out = layer(out, data.edge_index)
            if i != (len(self.__conv_layers) - 1):
                out = F.dropout(F.relu(out),
                                p=self.__dropout,
                                training=self.training)
        return out


class EdgeGAT(nn.Module):
    """
    Version of GAT that takes one-hot encoded edge attribute
    """

    def __init__(self,
                 node_attr_dim: int,
                 edge_attr_dim: int,
                 state_dim: int = 8,
                 num_heads: int = 8,
                 num_conv: int = 2,
                 out_dim: int = 1,
                 dropout: float = 0.2):

        super(EdgeGAT, self).__init__()

        self.__edge_attr_dim = edge_attr_dim
        __gat_kwargs = {
            'node_attr_dim': node_attr_dim,
            'state_dim': state_dim,
            'num_heads': num_heads,
            'num_conv': num_conv,
            'out_dim': out_dim,
            'dropout': dropout}

        self.__gat_nets = nn.ModuleList(
            [GAT(**__gat_kwargs) for _ in range(edge_attr_dim)])

    def forward(self, data: pyg_data.Data):

        out = []
        for i in range(self.__edge_attr_dim):
            # New graph that corresponds to the edge attributes
            _mask = data.edge_attr[:, i].byte()
            _edge_index = torch.masked_select(
                data.edge_index, mask=_mask).view(2, -1)
            _data = pyg_data.Data(x=data.x, edge_index=_edge_index)

            out.append(self.__gat_nets[i](_data))

        return torch.cat(tuple(out), dim=1)


class EdgeGATEncoder(nn.Module):

    def __init__(self,
                 node_attr_dim: int,
                 edge_attr_dim: int,
                 state_dim: int = 8,
                 num_heads: int = 8,
                 num_conv: int = 2,
                 out_dim: int = 1,
                 dropout: float = 0.2,
                 attention_pooling: bool = True):

        super(EdgeGATEncoder, self).__init__()

        self.__edge_gat = EdgeGAT(node_attr_dim=node_attr_dim,
                                  edge_attr_dim=edge_attr_dim,
                                  state_dim=state_dim,
                                  num_heads=num_heads,
                                  num_conv=num_conv,
                                  out_dim=state_dim,
                                  dropout=dropout)

        # Pooling layer is supposed to perform the following shape-shifting:
        #   From [num_nodes, node_attr_dim * edge_attr_dim]
        #   To [num_graphs, 2 * state_dim * edge_attr_dim]
        if attention_pooling:
            self.__pooling = pyg_nn.GlobalAttention(
                nn.Linear(state_dim * edge_attr_dim, 1),
                nn.Linear(state_dim * edge_attr_dim,
                          2 * state_dim * edge_attr_dim))
        else:
            self.__pooling = pyg_nn.Set2Set(state_dim * edge_attr_dim,
                                            processing_steps=3)

        self.__out_linear = nn.Sequential(
            nn.Linear(2 * state_dim * edge_attr_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, out_dim))

    def forward(self, data: pyg_data.Data):
        out = self.__edge_gat(data)
        out = self.__pooling(out, data.batch)
        return self.__out_linear(out)


# Testing segment for GAT with edge attributes and pooling layer
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

    model = EdgeGATEncoder(node_attr_dim=dataset.node_attr_dim,
                           edge_attr_dim=dataset.edge_attr_dim,
                           out_dim=len(TARGET_LIST)).to(device)

    model.train()
    data = next(iter(dataloader))

    print(f'The input batch data is {data}')
    data = data.to(device)
    print(f'The output shape is {model(data).shape}')
