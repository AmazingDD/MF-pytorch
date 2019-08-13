'''
@Author: Yu Di
@Date: 2019-08-08 14:43:33
@LastEditors: Yudi
@LastEditTime: 2019-08-13 16:10:21
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import torch

class SVDpp(torch.nn.Module):
    def __init__(self, params):
        super(SVDpp, self).__init__()
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

        self.yj = torch.nn.Embedding(self.num_items, self.latent_dim)

    def forward(self, user_idx, item_idx, Iu):
        '''
        Parameters
        ----------
        Iu: item set that user u interacted before
        '''
        user_vec = self.user_embedding(user_idx)
        u_impl_fdb = torch.zeros(user_idx.size(0), self.latent_dim)
        for j in Iu:
            j = torch.LongTensor([j])
            u_impl_fdb += self.yj(j)
        u_impl_fdb /= torch.FloatTensor([len(Iu)]).sqrt()
        user_vec += u_impl_fdb

        item_vec = self.item_embedding(item_idx)
        dot = torch.mul(user_vec, item_vec).sum(dim=1)
        rating = dot + self.mu + self.user_bias(user_idx).view(-1) + self.item_bias(item_idx).view(-1)

        return rating
