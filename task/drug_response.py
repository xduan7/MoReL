""" 
    File Name:          MoReL/drug_response.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               4/21/19
    Python Version:     3.5.4
    File Description:   

"""
import argparse

import sys
sys.path.extend(['/home/xduan7/Projects/MoReL'])
sys.path.extend(['/home/xduan7/Work/Projects/MoReL'])


def main():

    parser = argparse.ArgumentParser(
        description='Drug response prediction')

    # Arguments for drug feature type and model ###############################
    # drug_parsers = parser.add_subparsers()

    # Graph feature and model
    # drug_graph_parser = drug_parsers.add_parser('--drug_graph')

    parser.add_argument(
        '--drug_model_type', type=str, default='mpnn',
        help='type of convolutional graph model',
        choices=['mpnn', 'gcn', 'gat'])
    parser.add_argument(
        '--drug_pooling', type=str, default='set2set',
        help='global pooling layer for graph model',
        choices=['set2set', 'attention'])
    parser.add_argument(
        '--drug_state_dim', type=int, default=256,
        help='hidden state dimension for conv layers')
    parser.add_argument(
        '--drug_num_conv', type=int, default=3,
        help='number of convolution operations')
    parser.add_argument(
        '--drug_out_dim', type=int, default=128,
        help='output dimension of drug graph model')

    # SMILES string feature and model

    # Dragon 7 descriptor feature and model

    # Arguments for cell feature type and model ###############################
    parser.add_argument(
        '--cell_feature', type=str, default='rnaseq',
        help='feature for tumor cells',
        choices=['snp', 'rnaseq'])
    parser.add_argument(
        '--cell_state_dim', type=int, default=256,
        help='hidden state dimension for cell feature network')
    parser.add_argument(
        '--cell_num_layers', type=int, default=3,
        help='number of layers for cell feature network')
    parser.add_argument(
        '--cell_out_dim', type=int, default=128,
        help='output dimension of cell feature network')

    # Arguments for response prediction network ###############################
    parser.add_argument(
        '--resp_state_dim', type=int, default=256,
        help='hidden state dimension for drug response network')
    parser.add_argument('--resp_num_layers_per_block', type=int, default=2,
                        help='number of layers for drug response res block')
    parser.add_argument('--resp_num_blocks', type=int, default=2,
                        help='number of residual blocks for drug response')
    parser.add_argument('--resp_num_layers', type=int, default=2,
                        help='number of layers for drug response')
    parser.add_argument('--resp_dropout', type=float, default=0.0,
                        help='dropout of residual blocks for drug response')


    cell_parsers = parser.add_subparsers()

    # RNA sequence












