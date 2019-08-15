'''
@Author: Yu Di
@Date: 2019-08-15 17:40:21
@LastEditors: Yudi
@LastEditTime: 2019-08-15 17:44:50
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''

import torch

from utils.fm_layers import FeaturesLinear

class LR(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.linear = FeaturesLinear(params['field_dims'])

    def forward(self, x):
        '''
        @param x: Long tensor of size ``(batch_size, num_fields)``
        '''
        return torch.sigmoid(self.linear(x).squeeze(1))