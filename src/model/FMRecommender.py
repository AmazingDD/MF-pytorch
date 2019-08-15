'''
@Author: Yu Di
@Date: 2019-08-13 16:29:54
@LastEditors: Yudi
@LastEditTime: 2019-08-15 13:50:10
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import torch

from utils.fm_layers import FeaturesEmbedding, FeaturesLinear, FactorizationMachine

class FM(torch.nn.Module):
        super(FM, self).__init__()
        self.field_dims = params['field_dims']
        self.embed_dim = params['embed_dim']
        self.embedding = FeaturesEmbedding(self.field_dims, self.embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        '''
        x: Long tensor with size (batch_size, num_fields)
        '''
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))
        