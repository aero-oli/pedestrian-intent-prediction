
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.convolutional import stgcn
from torch_geometric_temporal.nn.recurrent import GConvGRU

filters = 32


class social_stgcn(torch.nn.Module):
    def __init__(self,n_stgcnn =1,n_txpcnn=1,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3):
        super(social_stgcn, self).__init__()
        self.n_stgcnn = n_stgcnn

        self.st_gcns = stgcn.STConv(input_feat, self.n_stgcnn, output_feat, (kernel_size, seq_len))

        self.recurrent = GConvGRU(output_feat, filters, 2)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.stgcnn(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.recurrent(h)
        h = F.relu(h)
        h = self.linear(h)

        return h