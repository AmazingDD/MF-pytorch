'''
@Author: Yu Di
@Date: 2019-08-09 14:04:38
@LastEditors: Yudi
@LastEditTime: 2019-08-13 15:19:32
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description:  this is a demo for biasMF recommendation
'''
import torch
from torch.utils.data import DataLoader, Dataset
from numpy.random import RandomState

class RateDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]
    
    def __len__(self):
        return self.user_tensor.size(0)

class NNTest(torch.nn.Module):
    def __init__(self, user_num=4, item_num=4, factors=10):
        super(NNTest, self).__init__()
        # self.user_embedding = torch.nn.Embedding(user_num, factors)
        # self.random_state = RandomState(1)

        self.user_embedding = torch.nn.Embedding(user_num, factors)
        self.user_embedding.weight.data = torch.randn(user_num, factors).float()

        self.item_embedding = torch.nn.Embedding(item_num, factors)
        self.item_embedding.weight.data = torch.randn(item_num, factors).float()

        self.user_bias = torch.nn.Embedding(user_num, 1)
        self.user_bias.weight.data = torch.zeros(user_num, 1).float()
        self.item_bias = torch.nn.Embedding(item_num, 1)
        self.item_bias.weight.data = torch.zeros(item_num, 1).float()

        self.global_mean = 0.0

    def forward(self, u, i):
        out = torch.mul(self.user_embedding(u), self.item_embedding(i)).sum(dim=1)
        out = out + self.user_bias(u) + self.item_bias(i) + self.global_mean

        return out

data = {(0,0): 4, 
        (0,1): 5, 
        (0,2): 3,
        (0,3): 4, 
        (1,0): 5, 
        (1,1): 3,
        (1,2): 4, 
        (1,3): 1, 
        (2,0): 3,
        (2,1): 2, 
        (2,2): 5, 
        (2,3): 5,
        (3,0): 4, 
        (3,1): 2, 
        (3,2): 3,
        (3,3): 1
        }
user_tensor = torch.LongTensor([key[0] for key in data.keys()])
item_tensor = torch.LongTensor([key[1] for key in data.keys()])
rating_tensor = torch.FloatTensor([val for val in data.values()])

model = NNTest()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

dataset = RateDataset(user_tensor, item_tensor, rating_tensor)
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

for epoch in range(30):
    for bid, batch in enumerate(train_loader):
        u, i, r = batch[0], batch[1], batch[2]
        r = r.float()
        # forward pass
        preds = model(u, i)
        loss = criterion(preds, r)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/30], Loss: {:.4f}'.format(epoch + 1, loss.item()))
    
    