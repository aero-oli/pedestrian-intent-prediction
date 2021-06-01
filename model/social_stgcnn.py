
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.attention import stgcn
from torch_geometric_temporal.nn.recurrent import GConvGRU

filters = 32


class social_stgcn(torch.nn.Module):
    def __init__(self,n_stgcnn =1,input_feat=2,output_feat=5,stgcn_kernel_size=3):
        super(social_stgcn, self).__init__()
        self.n_stgcnn = n_stgcnn

        self.st_gcns = stgcn.TemporalConv(in_channels=input_feat,
                                          out_channels=output_feat,
                                          kernel_size=stgcn_kernel_size)

        self.recurrent = GConvGRU(output_feat, filters, 2)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.stgcnn(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.recurrent(h)
        h = F.relu(h)
        h = self.linear(h)

        return F.log_softmax(h, dim=1)