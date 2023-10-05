import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE(nn.Module):
    def __init__(self, in_dim=784, latent_dim=20):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_dim = in_dim

        self.hdim = 512
        self.vdim = in_dim

        self.mu = nn.Linear(self.hdim, latent_dim)
        self.logvar = nn.Linear(self.hdim,latent_dim)

        self.l1e = nn.Sequential(nn.Linear(self.vdim, self.hdim), nn.LayerNorm(self.hdim), nn.ELU())
        self.l2e = nn.Sequential(nn.Linear(self.hdim, self.hdim), nn.LayerNorm(self.hdim), nn.ELU())
        self.l3e = nn.Sequential(nn.Linear(self.hdim, self.hdim), nn.LayerNorm(self.hdim), nn.ELU())

        self.l1d = nn.Sequential(nn.Linear(latent_dim, self.hdim), nn.LayerNorm(self.hdim), nn.ELU())
        self.l2d = nn.Sequential(nn.Linear(self.hdim, self.hdim), nn.LayerNorm(self.hdim), nn.ELU())
        self.l3d = nn.Sequential(nn.Linear(self.hdim, self.vdim), nn.LayerNorm(self.vdim), nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, x):
        x = self.l1e(x)
        x = x + self.l2e(x)
        x = x + self.l3e(x)

        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        z = self.l1d(z)
        z = z + self.l2d(z)
        x = self.l3d(z)
        return x

    def forward(self, x):
        z, mu, logvar = self.encode(x.view(-1, self.in_dim))
        rec_x = self.decode(z)
        return rec_x, mu, logvar

    def loss_function(self, rec_x, x, mu, logvar):
        x = (x + 1.)/2. # images are dynamically binarized to [-1,1]
        BCE = F.binary_cross_entropy(rec_x.view(-1, self.in_dim), x.view(-1, self.in_dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def NLLimportance(self, X, batchsz_importance):
        X = (X + 1.)/2. # images are dynamically binarized to [-1,1]

        batch_size = X.shape[0]
        ll = torch.Tensor(batch_size).to(X.device)

        with torch.no_grad():
            for i in range(batch_size):
                X_repeat = X[i].unsqueeze(0).repeat(batchsz_importance, 1)
                z, mu, logvar = self.encode(X_repeat.view(-1, self.in_dim))
                rec_x = self.decode(z)

                # p(x|z)
                logprob_QB =  (X_repeat * torch.log(rec_x) + (1 - X_repeat) * torch.log(1 - rec_x)).sum(-1)

                #p(z)
                distprior = torch.distributions.normal.Normal(0,1)
                logprior = distprior.log_prob(z).sum(-1)

                # q(z|x)
                dist = torch.distributions.normal.Normal(mu, torch.exp(0.5 * logvar))
                logprob_QF = dist.log_prob(z).sum(-1)

                ll[i] = torch.logsumexp(logprob_QB + logprior - logprob_QF, dim=0) - np.log(batchsz_importance)

        return -ll

    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim).to(device)
        x = self.decode(z)
        return x

    def recon(self,x):
        z,_,_ = self.encode(x)
        x = self.decode(z)
        x = 2 * torch.bernoulli(x) - 1
        return x
