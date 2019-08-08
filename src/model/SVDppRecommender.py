import torch

class SVDppRecommender(torch.nn.Module):
    def __init__(self, params):
        super(SVDppRecommender, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.latent_dim = params['latent_dim']
        self.mu = params['global_mean']

        self.user_embedding = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.latent_dim)

        self.user_bias = torch.zeros(self.num_users, requires_grad=True)
        self.item_bias = torch.zeros(self.num_items, requires_grad=True)

        self.yj = torch.randn((self.num_items, self.latent_dim), requires_grad=True)

        self.affine_output = torch.nn.Linear(self.latent_dim, 1)

    def forward(self, user_idx, item_idx, Iu):
        user_vec = self.user_embedding(user_idx)
        u_impl_fdb = torch.zeros(self.latent_dim)
        for j in Iu:
            u_impl_fdb += self.yj[j]
        u_impl_fdb /= torch.FloatTensor([len(Iu)]).sqrt()

        user_vec += u_impl_fdb
        item_vec = self.item_embedding(item_idx)
        dot = self.affine_output(torch.mul(user_vec, item_vec))
        rating = dot + self.mu + self.user_bias[user_idx] + self.item_bias[item_idx]

        return rating
