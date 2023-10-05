import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class IsingModel(nn.Module):
    def __init__(self, J, b, trainable=False):
        super(IsingModel, self).__init__()
        self.vdim = J.shape[0]
        self.trainable = trainable
        self.A = (J != 0).long()

        if trainable:
            self.J = nn.Parameter(torch.zeros_like(J))
            self.b = nn.Parameter(torch.zeros_like(b))
        else:
            self.J = J
            self.b = b

    def _get_J(self):
        if self.trainable:
            J = self.J + self.J.t()
            J = torch.where(self.A == 1, J, torch.zeros_like(J))
        else:
            J = self.J
        return J

    def get_adj(self):
        return self.A

    def get_energy(self, V):
        J = self._get_J()
        VJ = V @ J # (bs, vdim) @ (vdim, vdim) -> (bs, vdim)
        VJV = (VJ * V).sum(-1)  # (bs, vdim) -> (bs, )
        bV = (self.b[None, :] * V).sum(-1)
        return - VJV - bV  # (bs, )

    def forward(self, V, zero_one=True):
        if zero_one:
            V = 2 * V - 1
        return - self.get_energy(V)

    def get_energy_neighbor(self, V, ilist):
        J = self._get_J()
        E = -(V * J[ilist]).sum(1) * V.gather(1, ilist.unsqueeze(-1)).squeeze(-1) * 2 # batchsz
        E -= torch.gather(V * self.b, 1, ilist.unsqueeze(1)).squeeze(1)  # batchsz
        return E

    def get_conditional(self, V, i, invtemp=1):
        J = self._get_J()
        tmp = V @ J[:,i:i+1] # bs, 1
        return torch.sigmoid((4*tmp + 2*self.b[i]) * invtemp).squeeze() # bs

class FactorGraphModel(nn.Module):
    def __init__(self, graph, W1, b1, W2, b2, trainable=False): # K-nary interactions
        super(FactorGraphModel, self).__init__()
        self.graph = graph
        self.device = W1.device
        self.n_factors, self.K, self.hdim = W1.shape
        if trainable:
            self.W1 = nn.Parameter(0.01*torch.randn_like(W1))
            self.b1 = nn.Parameter(torch.zeros_like(b1))
            self.W2 = nn.Parameter(0.01*torch.randn_like(W2))
            self.b2 = nn.Parameter(torch.zeros_like(b2))
        else:
            self.W1, self.b1 = W1, b1
            self.W2, self.b2 = W2, b2

        assert self.K == 4
        if graph == "ladder":
            self.vdim = self.n_factors * 2 + 2
        else:
            self.vdim = (int(math.sqrt(self.n_factors)) + 1)**2

        self.factors, self.assign = self._create_factors_and_assignments() # self.vdim, x

    def _create_factors_and_assignments(self):
        if self.graph == "ladder":
            assign = torch.full((self.vdim, 2), self.n_factors)
        else:
            assign = torch.full((self.vdim, 4), self.n_factors)
        count = torch.zeros(self.vdim).long()
        factors = []
        for f in range(self.n_factors):
            if self.graph == "ladder":
                adj_list = [f*2, f*2+1, f*2+2, f*2+3]
            else:
                d = int(math.sqrt(self.vdim)) - 1
                adj_list = [f+f//d, f+f//d+1, f+f//d+d+1, f+f//d+d+2]
            factors.append(adj_list)
            for j in adj_list:
                assign[j, count[j]] = f
                count[j] += 1
        factors = torch.tensor(factors)
        return factors.long().to(self.device), assign.long().to(self.device)

    def _change_shape(self, V):
        batchsz = V.shape[0]
        factors = self.factors.unsqueeze(0).tile([batchsz, 1, 1]) # bs, n_factors, K
        V = V.gather(1, factors.reshape(batchsz, -1)) # bs, n_factors*K
        V = V.reshape(batchsz, self.n_factors, self.K).permute(1,0,2)
        return V

    def _forward(self, V):
        V = self._change_shape(V) # n_factors, bs, K
        V = V.bmm(self.W1) + self.b1 # n_factors, bs, hdim
        V = torch.tanh(V)
        V = V.bmm(self.W2) + self.b2 # n_factors, bs, 1
        V = V.squeeze(-1).t() # bs, n_factors
        return V

    def _forward_subset(self, V, i):
        f = self.assign[i]
        for j in range(f.shape[0]):
            if f[j] == self.n_factors:
                break
        f = f[:j] if f[j] == self.n_factors else f

        V = self._change_shape(V) # n_factors, bs, K
        V = V[f].bmm(self.W1[f]) + self.b1[f] # 2, bs, hdim
        V = torch.tanh(V)
        V = V.bmm(self.W2[f]) + self.b2[f] # 2, bs, 1
        V = V.squeeze(-1).t() # bs, 2
        return V

    def get_adj(self):
        fill = self.assign[:,0].unsqueeze(1).tile([1, self.assign.shape[1]])
        assign = torch.where(self.assign == self.n_factors, fill, self.assign)
        adj = torch.zeros(self.vdim, self.vdim)
        for i in range(assign.shape[1]):
            adj[torch.arange(adj.shape[0]), assign[:,i]] = 1.
        adj = (adj @ adj.t() > 0).long().fill_diagonal_(0.)
        return adj

    def get_energy(self, V):
        V = self._forward(V)
        return V.sum(1) # bs

    def forward(self, V, zero_one=True):
        if zero_one:
            V = 2 * V - 1
        return - self.get_energy(V)

    def get_energy_neighbor(self, V, ilist):
        # V: bs, vdim
        V = self._forward(V) # bs, n_factors
        V = torch.cat([V, torch.zeros_like(V[:,0:1])], 1) # bs, n_factors+1
        assign = self.assign[ilist] # bs, 2
        return V.gather(1, assign).sum(1)

    def get_conditional(self, V, i):
        # V: bs, vdim
        V_pos, V_neg = V.clone(), V.clone()
        V_pos[:,i], V_neg[:,i] = 1., -1.
        E_pos = self._forward_subset(V_pos, i).sum(1)
        E_neg = self._forward_subset(V_neg, i).sum(1)
        return torch.exp(-E_pos) / (torch.exp(-E_pos) + torch.exp(-E_neg))
