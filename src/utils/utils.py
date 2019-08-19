'''
@Author: Yu Di
@Date: 2019-08-08 13:36:45
@LastEditors: Yudi
@LastEditTime: 2019-08-19 17:00:05
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.model_selection import train_test_split

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
############ process for class FM ######################
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

########### MeF ###########
def load_ranking_data(path, test_size=0.2, header=['user_id', 'item_id', 'rating'], sep="\t"):
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    print(df.user_id.unique().size)
    print(df.item_id.unique().size)
    n_users = df.user_id.unique().size  # 943  #6040df.user_id.unique().shape[0]
    n_items = df.item_id.unique().size  # 1682 #3706#df.item_id.unique().shape[0]

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row, train_col, train_rating = [], [], []

    train_dict = {}
    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_dict[(u, i)] = 1
    print(len(train_dict))
    count = 0
    for u in range(n_users):
        for i in range(n_items):
            train_row.append(u)
            train_col.append(i)
            if (u, i) in train_dict.keys():
                count = count + 1
                train_rating.append(1)
            else:
                train_rating.append(0)

    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))
    all_items = set(np.arange(n_items))

    neg_user_item_matrix = {}
    train_user_item_matrix = []
    for u in range(n_users):
        neg_user_item_matrix[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))
        train_user_item_matrix.append(list(train_matrix.getrow(u).toarray()[0]))

    test_row, test_col, test_rating, unique_users = [], [], [], []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        unique_users.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))
    test_user_item_matrix = {}
    for u in range(n_users):
        test_user_item_matrix[u] = test_matrix.getrow(u).nonzero()[1]

    return train_matrix.todok(),  neg_user_item_matrix, test_matrix.todok(), test_user_item_matrix, n_users, n_items, set(unique_users)


def load_rating_data(path, header = ['user_id', 'item_id', 'rating', 'category'], test_size = 0.1, num_negatives= 0, sep="\t"):
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = df.user_id.unique().size
    n_items = df.item_id.unique().size
    print(n_users)
    print(n_items)

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row, train_col, train_rating, train_rating_1 = [], [], [], []

    train_dict = {}
    for line in  df.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        r = line[3]
        if (u,i) in test_data:
            continue
        train_dict[(u, i)] = r

    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])
        train_rating_1.append(1)
        for t in range(num_negatives):
            j = np.random.randint(n_items)
            while (u, j) in train_dict.keys():
                j = np.random.randint(n_items)
            train_row.append(u)
            train_col.append(j)
            train_rating.append(0)

    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))
    all_items = set(np.arange(n_items))
    train_user_item_matrix = []
    neg_user_item_matrix = {}
    for u in range(n_users):
        neg_user_item_matrix[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))
        train_user_item_matrix.append(list(train_matrix.getrow(u).toarray()[0]))

    test_row, test_col, test_rating, unique_users = [], [], [], []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        unique_users.append(line[1] - 1)
        test_rating.append(line[3])
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))
    test_user_item_matrix = {}
    for u in range(n_users):
        test_user_item_matrix[u] = test_matrix.getrow(u).nonzero()[1]

    return train_matrix.todok(), test_matrix.todok(), n_users, n_items, neg_user_item_matrix, test_user_item_matrix, unique_users
