'''
@Author: Yu Di
@Date: 2019-08-15 16:05:31
@LastEditors: Yudi
@LastEditTime: 2019-08-15 16:55:34
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Attentional Factorization Machine
'''
import torch
from utils.fm_layers import FeaturesEmbedding, FeaturesLinear, AttentionalFactorizationMachine

class AFM(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.num_fields = len(params['field_dims'])
        self.embedding = FeaturesEmbedding(params['field_dims'], params['embed_dim'])
        self.linear = FeaturesLinear(params['field_dims'])
        self.afm = AttentionalFactorizationMachine(params['embed_dim'], params['attn_size'], params['dropouts'])

    def forward(self, x):
        x = self.linear(x) + self.afm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))
