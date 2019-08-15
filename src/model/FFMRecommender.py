'''
@Author: Yu Di
@Date: 2019-08-15 11:23:00
@LastEditors: Yudi
@LastEditTime: 2019-08-15 16:07:25
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Field-Aware FM
'''
import torch
from utils.fm_layers import FeaturesLinear, FieldAwareFactorizationMachine

class FFM(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.linear = FeaturesLinear(params['field_dims'])
        self.ffm = FieldAwareFactorizationMachine(params['field_dims'], params['embed_dim'])

    def forward(self, x):
        '''
        x: Long tensor with size (batch_size, num_fields)
        '''
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        return torch.sigmoid(x.squeeze(1))
