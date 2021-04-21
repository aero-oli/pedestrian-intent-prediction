# import argparse
# import collections
# import torch
# import numpy as np
# import data_loader.data_loaders as module_data
# import model.loss as module_loss
# import model.metric as module_metric
# import model.model as module_arch
# from parse_config import ConfigParser
# from trainer import Trainer
# from utils import prepare_device
#
#
# # fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(SEED)
#
# def main(config):
#     logger = config.get_logger('train')
#
#     # setup data_loader instances
#     data_loader = config.init_obj('data_loader', module_data)
#     valid_data_loader = data_loader.split_validation()
#
#     # build model architecture, then print to console
#     model = config.init_obj('arch', module_arch)
#     logger.info(model)
#
#     # prepare for (multi-device) GPU training
#     device, device_ids = prepare_device(config['n_gpu'])
#     model = model.to(device)
#     if len(device_ids) > 1:
#         model = torch.nn.DataParallel(model, device_ids=device_ids)
#
#     # get function handles of loss and metrics
#     criterion = getattr(module_loss, config['loss'])
#     metrics = [getattr(module_metric, met) for met in config['metrics']]
#
#     # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
#     trainable_params = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
#     lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
#
#     trainer = Trainer(model, criterion, metrics, optimizer,
#                       config=config,
#                       device=device,
#                       data_loader=data_loader,
#                       valid_data_loader=valid_data_loader,
#                       lr_scheduler=lr_scheduler)
#
#     trainer.train()
#
#
# if __name__ == '__main__':
#     args = argparse.ArgumentParser(description='PyTorch Template')
#     args.add_argument('-c', '--config', default=None, type=str,
#                       help='config file path (default: None)')
#     args.add_argument('-r', '--resume', default=None, type=str,
#                       help='path to latest checkpoint (default: None)')
#     args.add_argument('-d', '--device', default=None, type=str,
#                       help='indices of GPUs to enable (default: all)')
#
#     # custom cli options to modify configuration from default values given in json file.
#     CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
#     options = [
#         CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
#         CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
#     ]
#     config = ConfigParser.from_args(args, options)
#     main(config)



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
    print("Start!")
    Ep_Forecasting()
    Web_Traf_Pred()
    print("End!")
