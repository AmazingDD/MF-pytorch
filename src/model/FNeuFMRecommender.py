'''
@Author: Yu Di
@Date: 2019-08-15 18:04:26
@LastEditors: Yudi
@LastEditTime: 2019-08-16 10:18:11
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: FieldAwareNeuralFactorizationMachine
'''
import torch
from utils.fm_layers import FieldAwareFactorizationMachine, MultiLayerPerceptron, FeaturesEmbedding
from LRRecommender import LR

class FNeuFM(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        field_dims = params['field_dims']
        embed_dim = params['embed_dim']
        dropouts = params['dropouts']
        mlp_dims = params['mlp_dims']
        dropouts = params['dropouts']

        self.embed_layer = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = LR(field_dims)
        self.ffm = torch.nn.Sequential(FieldAwareFactorizationMachine(field_dims, embed_dim), 
                                       torch.nn.BatchNorm1d(embed_dim), 
                                       torch.nn.Dropout(dropouts[0]))
        self.ffm_output_dim = len(field_dims) * (len(field_dims) - 1) // 2 * embed_dim
        self.mlp = MultiLayerPerceptron(self.ffm_output_dim, mlp_dims, dropouts[1])

    def forward(self, x):
        cross_term = self.ffm(self.embed_layer(x))
        x = self.linear(x) + self.mlp(cross_term.view(-1, self.ffm_output_dim))
        return torch.sigmoid(x.squeeze(1))
