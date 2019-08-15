'''
@Author: Yu Di
@Date: 2019-08-15 17:36:32
@LastEditors: Yudi
@LastEditTime: 2019-08-15 18:00:13
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Neural Factorization Machine
'''
import torch

from LRRecommender import LR
from utils.fm_layers import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron

class NeuFM(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = FeaturesEmbedding(params['field_dims'], params['embed_dim'])
        self.linear = LR(params['field_dims'])
        self.fm = torch.nn.Sequential(FactorizationMachine(reduce_sum=False), 
                                      torch.nn.BatchNorm1d(params['embed_dim']), # Bi-Interaction layer
                                      torch.nn.Dropout(params['dropouts'][0]))
        self.mlp = MultiLayerPerceptron(params['embed_dim'], params['mlp_dims'], params['dropouts'][1])

    def forward(self, x):
        cross_term = self.fm(self.embedding(x))
        x = self.linear(x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))
