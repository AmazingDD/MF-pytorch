'''
@Author: Yu Di
@Date: 2019-08-15 14:44:51
@LastEditors: Yudi
@LastEditTime: 2019-08-15 15:06:23
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import torch
from utils.fm_layers import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron, CompressedInteractionNetwork

class XDFM(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = FeaturesEmbedding(params['field_dims'], params['embed_dim'])
        self.embed_output_dim = len(params['field_dims']) * params['embed_dim']
        self.cin = CompressedInteractionNetwork(len(params['field_dims']), params['cross_layer_sizes'], params['split_half'])
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, params['mlp_dims'], params['dropout'])
        self.linear = FeaturesLinear(params['field_dims'])

    def forward(self, x):
        embed_x = self.embedding(x)
        x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))
