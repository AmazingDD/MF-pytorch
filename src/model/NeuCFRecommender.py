import torch
from GMFRecommender import GMF
from MLPRecommender import MLP

class NeuCF(torch.nn.Module):
    def __init__(self, params):
        super(NeuCF, self).__init__()
        self.params = params
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.latent_dim_mf = params['latent_dim_mf']
        self.latent_dim_mlp = params['latent_dim_mlp']

        self.user_embedding_mf = torch.nn.Embedding(self.num_users, self.latent_dim_mf)
        self.user_embedding_mlp = torch.nn.Embedding(self.num_users, self.latent_dim_mlp)
        self.item_embedding_mf = torch.nn.Embedding(self.num_items, self.latent_dim_mf)
        self.item_embedding_mlp = torch.nn.Embedding(self.num_items, self.latent_dim_mlp)

        self.fc_layers = torch.nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(params['layers'][:-1], params['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        self.relu = torch.nn.ReLU()

        self.affine_output = torch.nn.Linear(params['layers'][-1] + params['latent_dim_mf'], 1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mf = self.user_embedding_mf(user_indices)
        item_embedding_mf = self.item_embedding_mf(item_indices)
        user_embedding_mlp = self.user_embedding_mlp(user_indices)
        item_embedding_mlp = self.item_embedding_mlp(item_indices)

        mlp_vec = torch.cat([user_embedding_mlp, item_embedding_mlp], -1)
        mf_vec = torch.mul(user_embedding_mf, item_embedding_mf)
        for _, layer in enumerate(self.fc_layers):
            mlp_vec = layer(mlp_vec)
            mlp_vec = self.relu(mlp_vec)

        vec = torch.cat([mlp_vec, mf_vec], dim=-1)
        out = self.affine_output(vec)
        rating = self.logistic(out)

        return rating

    def load_pretrain(self, mlp_dirs, mf_dirs):
        params = self.params
        params['latent_dim'] = params['latent_dim_mlp']
        mlp = MLP(params)
        state_dict = torch.load(mlp_dirs)
        mlp.load_state_dict(state_dict)

        self.item_embedding_mlp.weight.data = mlp.item_embedding.weight.data
        self.user_embedding_mlp.weight.data = mlp.user_embedding.weight.data
        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp.fc_layers[idx].weight.data

        params['latent_dim'] = params['latent_dim_mf']
        gmf = GMF(params)
        state_dict = torch.load(mf_dirs)
        gmf.load_state_dict(state_dict)

        self.user_embedding_mf.weight.data = gmf.user_embedding.weight.data
        self.item_embedding_mf.weight.data = gmf.item_embedding.weight.data

        self.affine_output.weight.data = 0.5 * torch.cat([mlp.affine_output.weight.data, gmf.affine_output.weight.data], -1)
        self.affine_output.bias.data = 0.5 * (mlp.affine_output.bias.data + gmf.affine_output.bias.data)
