'''
@Author: Yu Di
@Date: 2019-08-15 17:01:19
@LastEditors: Yudi
@LastEditTime: 2019-08-15 17:24:39
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Wide and Deep
'''
import torch
from utils.fm_layers import FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron

class WaD(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.linear = FeaturesLinear(params['field_dims'])
        self.embedding = FeaturesEmbedding(params['field_dims'], params['embed_dim'])
        self.embed_output_dim = len(params['field_dims']) * params['embed_dim']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, params['mlp_dims'], dropout)

    def forward(self, x):
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))
