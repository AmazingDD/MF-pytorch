import torch

# # Device configuration
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# images = images.to(device)
# labels = labels.to(device)

class GMF(torch.nn.Module):
    def __init__(self, params):
        super(GMF, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.latent_dim = params['latent_dim']

        self.user_embedding = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.latent_dim)

        self.affine_output = torch.nn.Linear(self.latent_dim, 1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        dot = torch.mul(user_vec, item_vec)
        out = self.affine_output(dot)
        rating = self.logistic(out)

        return rating


# gmf = GMFRecommender(params)
# torch.save(gmf.state_dict(), dirs)