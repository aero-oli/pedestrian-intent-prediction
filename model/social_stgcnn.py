
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.attention.stgcn import STConv
from torch_geometric_temporal.nn.recurrent import GConvGRU

filters = 32


class social_stgcn(torch.nn.Module):
    def __init__(self, input_feat=2, hidden_channels=5, output_feat=5,
                 stgcn_kernel_size=3, num_nodes=1, K=1):
        super(social_stgcn, self).__init__()

        self.input_feat = input_feat

        self.st_gcns = STConv(in_channels=input_feat,
                              hidden_channels=hidden_channels,
                              out_channels=output_feat,
                              kernel_size=stgcn_kernel_size,
                              num_nodes=num_nodes,
                              K=K)
        self.gcn1 = torch.nn.ModuleList([GCNConv(in_channels=input_feat,
                                                 out_channels=output_feat)])
        self.gcn2 = torch.nn.ModuleList([GCNConv(in_channels=output_feat,
                                                 out_channels=1)])

        self.hidden = [GCNConv(in_channels=output_feat,
                               out_channels=output_feat)
                       for _ in range(hidden_channels-1)]

        self.hidden = torch.nn.ModuleList(self.hidden)
        self.recurrent = GConvGRU(input_feat, filters, 2)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # print(x)
        # print(edge_index)

        x = torch.cat([x,
                       torch.zeros(size=(self.input_feat-x.size()[0],
                                         x.size()[1]))], 0)
        # # print(x)
        # # h = self.st_gcns(x, edge_index)
        #
        # h = self.gcn[0](x=x, edge_index=edge_index)
        #
        # for idx, gcn in enumerate(self.hidden):
        #     h = gcn(h)
        #
        # h = F.relu(h)
        # h = self.recurrent(h)
        # h = F.relu(h)
        # h = self.linear(h)

        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gcn2(x, edge_index)

        return F.log_softmax(x, dim=1)