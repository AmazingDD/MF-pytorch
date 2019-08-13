'''
@Author: Yu Di
@Date: 2019-08-08 14:18:50
@LastEditors: Yudi
@LastEditTime: 2019-08-13 16:08:18
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import torch

class BiasMF(torch.nn.Module):
    def __init__(self, params):
        super(BiasMF, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.latent_dim = params['latent_dim']
        self.mu = params['global_mean']

        self.user_embedding = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.latent_dim)

        self.user_bias = torch.nn.Embedding(self.num_users, 1)
        self.user_bias.weight.data = torch.zeros(self.num_users, 1).float()
        self.item_bias = torch.nn.Embedding(self.num_items, 1)
        self.item_bias.weight.data = torch.zeros(self.num_items, 1).float()

    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        dot = torch.mul(user_vec, item_vec).sum(dim=1)

        rating = dot + self.mu + self.user_bias(user_indices).view(-1) + self.item_bias(item_indices).view(-1) + self.mu

        return rating
