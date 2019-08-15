'''
@Author: Yu Di
@Date: 2019-08-08 13:36:45
@LastEditors: Yudi
@LastEditTime: 2019-08-15 14:11:42
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                                          lr=params['adam_lr'],
                                                          weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer

def use_criterion(params):
    if params['crit'] == 'explicit':
        return torch.nn.MSELoss()
    elif params['crit'] == 'implict':
        return torch.nn.BCELoss()

class RateDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]
    
    def __len__(self):
        return self.user_tensor.size(0)

class Movilens1MDataset(Dataset):
    def __init__(self, sep='::'):
        data = pd.read_csv('data/ml-1m/movies.dat', sep=sep, engine='python').to_numpy()[:, :3]
        self.items = data[:, :2].astype(int)
        self.targets = self._preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def _preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target
