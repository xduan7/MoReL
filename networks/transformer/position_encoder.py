""" 
    File Name:          MoReL/position_encoder.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/7/19
    Python Version:     3.5.4
    File Description:   

"""
import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionEncoder(nn.Module):

    def __init__(self,
                 seq_length: int,
                 emb_dim: int,
                 emb_scale: float = None,
                 dropout: float = 0.0,
                 base_feq: float = 8.0):

        super().__init__()

        self.__emb_dim = emb_dim

        # The embedding scale prevents embedded vector get diminished for
        # adding positional encoding
        self.__emb_scale = emb_scale if emb_scale \
            else math.sqrt(self.__emb_dim)

        self.__dropout = nn.Dropout(p=dropout)

        # This is a very generic way of creating PE
        pos_enc_mat = torch.zeros(seq_length, emb_dim)

        for pos in range(seq_length):
            for i in range(0, emb_dim, 2):
                pos_enc_mat[pos, i] = math.sin(
                    pos / (base_feq ** ((2 * i) / emb_dim)))
                pos_enc_mat[pos, i + 1] = math.cos(
                    pos / (base_feq ** ((2 * (i + 1)) / emb_dim)))

        # Register positional encoding matrix
        # Note that this serves as constant (not learnable)
        # Use register_buffer (returns tensor) instead of nn.Parameter()
        self.register_buffer('pos_enc_mat', pos_enc_mat.unsqueeze(0))

    def forward(self, x):

        # Input size: (batch_size, seq_length, emb_dim)
        # Output size: (batch_size, seq_length, emb_dim)

        # Scaling embedded input (could do this in the embedding layer)
        x = x * self.__emb_scale

        # Add positional variable to embedding
        x += Variable(self.pos_enc_mat[:, :x.size(1)],
                      requires_grad=False)

        return self.__dropout(x)


# Test out the positional encoding layer by showing the matrix (curves)
if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(24, 8))

    seq_len = 256
    dim = 32
    freq = 16.0

    batch_input = Variable(torch.zeros(1, seq_len, dim)).cuda()
    pe = PositionEncoder(seq_len, dim, base_feq=freq).cuda()
    batch_output = pe.forward(batch_input).cpu()
    plt.plot(np.arange(seq_len), batch_output[0, :, :].data.numpy())

    plt.show()
