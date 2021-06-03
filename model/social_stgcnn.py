
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.attention.stgcn import STConv
from torch_geometric_temporal.nn.recurrent import GConvGRU

filters = 32


class social_stgcn(torch.nn.Module):
    def __init__(self, input_feat=2, hidden_channels=1, output_feat=5,
                 stgcn_kernel_size=3, num_nodes=1, K=1):
        super(social_stgcn, self).__init__()

        self.st_gcns = STConv(in_channels=input_feat,
                              hidden_channels=hidden_channels,
                              out_channels=output_feat,
                              kernel_size=stgcn_kernel_size,
                              num_nodes=num_nodes,
                              K=K)

        self.recurrent = GConvGRU(output_feat, filters, 2)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.st_gcns(x, edge_index)
        h = F.relu(h)
        h = self.recurrent(h)
        h = F.relu(h)
        h = self.linear(h)

        return F.log_softmax(h, dim=1)