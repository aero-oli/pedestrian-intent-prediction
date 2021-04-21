
def Ep_Forecasting():
    from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
    from torch_geometric_temporal.signal import temporal_signal_split

    loader = ChickenpoxDatasetLoader()

    dataset = loader.get_dataset()

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

    import torch
    import torch.nn.functional as F
    from torch_geometric_temporal.nn.recurrent import DCRNN

    class RecurrentGCN(torch.nn.Module):
        def __init__(self, node_features):
            super(RecurrentGCN, self).__init__()
            self.recurrent = DCRNN(node_features, 32, 1)
            self.linear = torch.nn.Linear(32, 1)

        def forward(self, x, edge_index, edge_weight):
            h = self.recurrent(x, edge_index, edge_weight)
            h = F.relu(h)
            h = self.linear(h)
            return h

    from tqdm import tqdm

    model = RecurrentGCN(node_features=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()

    for epoch in tqdm(range(200)):
        cost = 0
        for time, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
        cost = cost / (time + 1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    cost = 0
    for time, snapshot in enumerate(test_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
    cost = cost / (time + 1)
    cost = cost.item()
    print("MSE: {:.4f}".format(cost))


def Web_Traf_Pred():
    from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
    from torch_geometric_temporal.signal import temporal_signal_split

    loader = WikiMathsDatasetLoader()

    dataset = loader.get_dataset(lags=14)

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.5)

    import torch
    import torch.nn.functional as F
    from torch_geometric_temporal.nn.recurrent import GConvGRU

    class RecurrentGCN(torch.nn.Module):
        def __init__(self, node_features, filters):
            super(RecurrentGCN, self).__init__()
            self.recurrent = GConvGRU(node_features, filters, 2)
            self.linear = torch.nn.Linear(filters, 1)

        def forward(self, x, edge_index, edge_weight):
            h = self.recurrent(x, edge_index, edge_weight)
            h = F.relu(h)
            h = self.linear(h)
            return h

    from tqdm import tqdm

    model = RecurrentGCN(node_features=14, filters=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()

    for epoch in tqdm(range(50)):
        for time, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = torch.mean((y_hat - snapshot.y) ** 2)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    cost = 0
    for time, snapshot in enumerate(test_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
    cost = cost / (time + 1)
    cost = cost.item()
    print("MSE: {:.4f}".format(cost))

if __name__ == '__main__':
    print('Start!')
    Ep_Forecasting()
    Web_Traf_Pred()
    print('End!')
