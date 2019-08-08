import torch

class BiasMFRecommender(torch.nn.Module):
    def __init__(self, params):
        super(BiasMFRecommender, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.latent_dim = params['latent_dim']
        self.mu = torch.tensor(params['global_mean')

        self.user_embedding = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.latent_dim)

        self.user_bias = torch.zeros(self.num_users, requires_grad=True)
        self.item_bias = torch.zeros(self.num_items, requires_grad=True)

        self.affine_output = torch.nn.Linear(self.latent_dim, 1)

    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        dot = torch.mul(user_vec, item_vec)
        out = self.affine_output(dot)
        rating = out + self.mu + self.user_bias[user_indices] + self.item_bias[item_indices]

        return rating
