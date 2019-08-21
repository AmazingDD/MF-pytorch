'''
@Author: Yu Di
@Date: 2019-08-21 10:21:14
@LastEditors: Yudi
@LastEditTime: 2019-08-21 15:30:17
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import torch

class GRU4REC(torch.nn.Module):
    def __init__(self, params, final_act='tanh'):
        super(GRU4REC, self).__init__()
        self.input_size = params['input_size']
        self.hidden_size = params['hidden_size']
        self.output_size = params['output_size']
        self.num_layers = params['num_layers']
        self.dropout_hidden = params['dropout_hidden']
        self.dropout_input = params['dropout_input']
        self.embedding_dim = params['embedding_dim']
        self.batch_size = params['batch_size']
        self.use_cuda = params['use_cuda']
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.onehot_buffer = self.init_emb()
        self.h2o = torch.nn.Linear(self.hidden_size, self.output_size)

        self.create_fnl_activation(final_act)

        if self.embedding_dim != -1:
            self.look_up = torch.nn.Embedding(self.input_size, self.embedding_dim)
            self.gru = torch.nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        else:
            self.gru = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        self = self.to(self.device)

    def create_fnl_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = torch.nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = torch.nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = torch.nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))

    def forward(self, input, hidden):
        '''
         input (B,): a batch of item indices from a session-parallel mini-batch.
         target (B,): torch.LongTensor of next item indices from a session-parallel mini-batch.

         logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
         hidden: GRU hidden state
        '''
        if self.embedding_dim != -1:
            embedded = self.onehot_encode(input)
            if self.training and self.dropout_input > 0:
                embedded = self.embedding_dropout(embedded)
            embedded = embedded.unsqueeze(0)
        else:
            embedded = input.unsqueeze(0)
            embedded = self.look_up(embedded)
        
        output, hidden = self.gru(embedded, hidden) # (num_layer, B, H)
        output = output.view(-1, output.size(-1))  # (B, H) at last layer
        logit = self.final_activation(output)

        return logit, hidden

    def init_emb(self):
        onehot_buffer = torch.FloatTensor(self.batch_size, self.output_size)
        onehot_buffer = onehot_buffer.to(self.device)

        return onehot_buffer

    def onehot_encode(self, input):
        self.onehot_buffer.zero_()
        index = input.view(-1, 1)
        one_hot = self.onehot_buffer.scatter_(1, index, 1) # (B, C)

        return one_hot

    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), 1).fill_(1 - self.dropout_input) # (B, 1)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input) # (B, C)
        mask = mask.to(self.device)
        input = input * mask

        return input

    def init_hidden(self):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return h0
