'''
@Author: Yu Di
@Date: 2019-08-16 10:36:10
@LastEditors: Yudi
@LastEditTime: 2019-08-16 11:27:33
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Automatic Feature Interaction
'''
import torch
import torch.nn.functional as F

from utils.fm_layers import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class AFI(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        field_dims = params['field_dims']
        embed_dim = params['embed_dim']
        mlp_dims = params['mlp_dims']
        dropouts = params['dropouts']
        num_heads = params['num_heads']
        num_layers = params['num_layers']

        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[1])
        self.self_attns = torch.nn.ModuleList([torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)])
        self.attn_fc = torch.nn.Linear(self.embed_output_dim, 1)

    def forward(self, x):
        embed_x = self.embedding(x)
        cross_term = embed_x.transpose(0, 1)
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)
        cross_term = F.relu(cross_term).contiguous().view(-1, self.embed_output_dim)
        x = self.linear(x) + self.attn_fc(cross_term) + self.mlp(embed_x.view(-1, self.embed_output_dim))

        return torch.sigmoid(x.squeeze(1))
