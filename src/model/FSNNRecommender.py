'''
@Author: Yu Di
@Date: 2019-08-15 13:32:58
@LastEditors: Yudi
@LastEditTime: 2019-08-15 13:47:12
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import torch
from utils.fm_layers import FeaturesEmbedding, MultiLayerPerceptron

class FSNN(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = FeaturesEmbedding(params['field_dims'], params['embed_dim'])
        self.embed_output_dim = len(params['field_dims']) * params['embed_dim']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, params['mlp_dims'], params['dropout'])

    def forward(self, x):
        '''
        x: Long tensor with size (batch_size, num_fields)
        '''
        embed_x = self.embedding(x)
        x = self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))
