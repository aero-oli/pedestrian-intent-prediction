
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, Sequential
from torch_geometric_temporal.nn.recurrent import GCLSTM, GConvLSTM

filters = 32


class social_stgcn(torch.nn.Module):
    def __init__(self, input_feat=2, Conv_outputs=[5], LSTM_output=[5],
                 K=1, linear_output=3):
        super(social_stgcn, self).__init__()

        self.input_feat = input_feat
        self.Conv_outputs = Conv_outputs
        self.LSTM_output = LSTM_output
        self.linear_output = linear_output
        self.K = K

        self.gcn = Sequential('x, edge_index, h', [
                                  (GCNConv(in_channels=self.input_feat,
                                           out_channels=self.Conv_outputs[0],
                                           improved=True), 'x, edge_index -> x'),
                                  nn.ReLU(),
                                  (GCNConv(in_channels=self.Conv_outputs[0],
                                           out_channels=self.Conv_outputs[1],
                                           improved=True), 'x, edge_index -> x'),
                                  # (GCNConv(in_channels=self.input_feat,
                                  #          out_channels=self.input_feat,
                                  #          improved=True,
                                  #          normalize=True,
                                  #          bias=True), 'x, edge_index -> x'),
                                  # (GCNConv(in_channels=self.input_feat,
                                  #          out_channels=self.output_feat,
                                  #          improved=True,
                                  #          normalize=True,
                                  #          bias=True), 'x, edge_index -> x'),
                                  nn.ReLU(),
                                  (GConvLSTM(in_channels=self.Conv_outputs[1],
                                             out_channels=self.Conv_outputs[1],
                                             K=K, normalization="sym", bias=True), 'x, edge_index -> h, _'),
                                  nn.ReLU(), "h -> h",

                                  (GConvLSTM(in_channels=self.Conv_outputs[1],
                                             out_channels=self.Conv_outputs[1],
                                             K=K, normalization="sym", bias=True), 'h, edge_index -> h, _'),
                                  nn.ReLU(), "h -> h",

                                  (GConvLSTM(in_channels=self.Conv_outputs[1],
                                             out_channels=self.LSTM_output[0],
                                             K=K, normalization="sym", bias=True), 'h, edge_index -> _, c'),
                                    # TO BE TESTED LATER
                                  # (GCLSTM(in_channels=self.Conv_outputs[1],
                                  #         out_channels=self.LSTM_output[0],
                                  #         K=K, normalization="sym", bias=True), 'x, edge_index -> h, c'),
                                  #
                                  # (GCLSTM(in_channels=self.LSTM_output[0],
                                  #         out_channels=self.LSTM_output[1],
                                  #         K=K, normalization="sym", bias=True), 'h, edge_index, c -> h, c'),
                                  #
                                  # (GCLSTM(in_channels=self.LSTM_output[1],
                                  #         out_channels=self.LSTM_output[2],
                                  #         K=K, normalization="sym", bias=True), 'h, edge_index, c -> h, _'),
                                  (nn.ReLU(), "c -> c"),
                                  (nn.Linear(in_features=self.LSTM_output[0],
                                             out_features=self.linear_output), "c -> x")])


    def forward(self, data, device):
        x, edge_index, y = data.x.cuda(), \
                                  data.edge_index.cuda(), \
                                  data.y.cuda()

        x = torch.cat([x,
                       torch.zeros(size=(self.input_feat-x.size()[0],
                                         x.size()[1]), device=device)], 0)
        x = torch.cat([x,
                       torch.zeros(size=(x.size()[0],
                                         self.input_feat - x.size()[1]), device=device)], 1)
        h = torch.zeros(size=edge_index.shape)
        # return F.log_softmax(x, dim=1)
        return self.gcn(x, edge_index, h)
        # return torch.round(self.gcn(x, edge_index))