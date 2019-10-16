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

class FeaturesLinear(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
    
    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias

class LR(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.linear = FeaturesLinear(params['field_dims'])

    def forward(self, x):
        '''
        @param x: Long tensor of size ``(batch_size, num_fields)``
        '''
        return torch.sigmoid(self.linear(x).squeeze(1))