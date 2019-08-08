import torch
from GMFRecommender import GMFRecommender

class MLPRecommender(torch.nn.Module):
    def __init__(self, params):
        super(MLPRecommender, self).__init__()
        self.params = params
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.latent_dim = params['latent_dim']

        self.user_embedding = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(params['layers'][:-1], params['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.relu = torch.nn.ReLU()
        self.affine_output = torch.nn.Linear(params['layers'][-1], 1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)

        vec = torch.cat([user_vec, item_vec], dim=-1)

        for _, layer in enumerate(self.fc_layers):
            vec = layer(vec)
            vec = self.relu(vec)
            vec = torch.nn.Dropout(p=0.5)(vec)
        out = self.affine_output(vec)
        rating = self.logistic(out)

        return rating

    def load_pretrain(self, dirs):
        params = self.params
        gmf = GMFRecommender(params)
        
        state_dict = torch.load(dirs)
        gmf.load_state_dict(state_dict)
        
        self.user_embedding.weight.data = gmf.user_embedding.weight.data
        self.item_embedding.weight.data = gmf.item_embedding.weight.data


# mlp = MLPRecommender(params)
# mlp.load_pretrain(dirs)