'''
@Author: Yu Di
@Date: 2019-08-21 15:44:56
@LastEditors: Yudi
@LastEditTime: 2019-08-22 14:45:34
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Session-based Recommendation with Graph Neural Networks
'''
import math
import torch
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = torch.nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = torch.nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = torch.nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = torch.nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = torch.nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = torch.nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]:2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(inputs, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + i_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class SessionGraph(torch.nn.Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hidden_size
        self.n_node = n_node
        self.batch_size = opt.batch_size
        self.nonhybrid = opt.nonhybrid
        self.embedding = torch.nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = torch.nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = torch.nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1] # batch_size * latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size * 1 * latent_size
        q2 = self.linear_two(hidden)  # batch_size * seq_length * latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0, -1, 1]).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]    # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden

def transfer_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable



