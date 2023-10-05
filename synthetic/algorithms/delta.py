import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnet(nn.Module):
    def __init__(self, vdim, hdim, act):
        super(Qnet, self).__init__()
        self.vdim = vdim
        self.hdim = hdim
        if act == 'relu':
            activation = nn.ReLU
        elif act == 'elu':
            activation = nn.ELU
        else:
            activation = nn.Tanh

        self.layers = nn.Sequential(
            nn.Linear(self.vdim, self.hdim),
            nn.LayerNorm(self.hdim),
            activation(),
            nn.Linear(self.hdim, self.hdim),
            nn.LayerNorm(self.hdim),
            activation(),
            nn.Linear(self.hdim, self.hdim),
            nn.LayerNorm(self.hdim),
            activation(),
        )

        self.W = nn.Parameter(0.01*torch.randn(self.vdim, self.hdim, 1))
        self.b = nn.Parameter(torch.zeros(self.vdim, 1, 1))
        self.marginals = nn.Parameter(torch.zeros(self.vdim))
        self.bceloss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, V, ilist, temp=1):
        # V: batchsz, vdim
        x = self.layers(V).unsqueeze(1) # batchsz, 1, hdim
        W = self.W[ilist] # batchsz, hdim, 1
        b = self.b[ilist] # batchsz, 1, 1
        x = (x.bmm(W) + b).squeeze() # batchsz
        x = torch.where(V.abs().sum(-1) == 0, self.marginals[ilist], x)
        return x / temp

    def log_prob(self, V_pa, ilist, V):
        logits = self.forward(V_pa, ilist, temp=1) # batchsz
        labels = torch.gather((V+1)/2, -1, ilist.unsqueeze(-1)).squeeze(-1) # batchsz
        return -self.bceloss(logits, labels)

    def sample(self, V_pa, ilist, temp=1):
        p = torch.sigmoid(self.forward(V_pa, ilist, temp=temp))
        return 2 * torch.bernoulli(p).float() - 1

class DeltaAI(nn.Module):
    def __init__(self, args, device, model, pgm):
        super(DeltaAI, self).__init__()

        self.vdim = args.vdim
        self.hdim = args.hdim
        self.batchsz = args.batchsz
        self.batchsz_nll = args.batchsz_nll
        self.batchsz_proc = 1000
        self.device = device
        self.alg = args.alg

        self.model = model
        self.pgm = pgm

        self.Qnet = Qnet(self.vdim, self.hdim, args.act)
        self.factor = 1

    def set_factor(self, factor):
        self.factor = factor

    def _get_conditionals(self, K_pa, K_ch, V, ilist):

        with torch.no_grad():
            instance_pa_idx, pa_idx = self.pgm.get_idx_from_K(K_pa, ilist)
            instance_ch_idx, ch_idx = self.pgm.get_idx_from_K(K_ch, ilist)

            if self.alg == "rand":
                K_pa_ch = torch.index_select(K_pa, 0, instance_ch_idx)

        # logQ_Vi_pai (or logQ_pVi_pai)
        V_pa = torch.zeros_like(V)
        V_pa[instance_pa_idx, pa_idx] = V[instance_pa_idx, pa_idx] # batchsz, vdim
        logQ_Vi_pai = self.Qnet.log_prob(V_pa, ilist, V)

        # batch itself to avoid memory overflow
        batch_idx = torch.split(torch.arange(ch_idx.shape[0]).to(self.device), self.batchsz_proc)

        sum_logQ_Vj_paj = torch.zeros_like(V)
        for i_batch, idx in enumerate(batch_idx):

            with torch.no_grad():
                itself_batch = ch_idx[idx]
                instance_ch_idx_batch = instance_ch_idx[idx]

                if self.alg == "rand":
                    K_pa_batch = K_pa_ch[idx]
                else:
                    K_pa_batch = K_pa

                instance_idx, pa_idx = self.pgm.get_idx_from_K(K_pa_batch, itself_batch)

            V_pa = torch.zeros(itself_batch.shape[0], self.vdim).to(self.device)
            V_ch_i = torch.index_select(V, 0, instance_ch_idx_batch)
            V_pa_j = torch.index_select(V_ch_i, 0, instance_idx)
            V_pa[instance_idx, pa_idx] = torch.gather(V_pa_j, 1, pa_idx.unsqueeze(1)).squeeze(1)

            ll = self.Qnet.log_prob(V_pa, itself_batch, V_ch_i)
            sum_logQ_Vj_paj[instance_ch_idx_batch, itself_batch] = ll

        sum_logQ_Vj_paj = sum_logQ_Vj_paj.sum(1)

        return logQ_Vi_pai, sum_logQ_Vj_paj

    def get_loss(self, V, ilist):
        if self.alg == "fixed":
            K_pa, K_ch = self.pgm.K_pa_full, self.pgm.K_ch_full
        else:
            K_pa, K_ch = self.pgm.K_pa, self.pgm.K_ch

        # perturb
        V_ = torch.where(F.one_hot(ilist, num_classes=self.vdim) > 0, -V, V)

        # directed part
        logQ_Vi_pai, sum_logQ_Vj_paj = self._get_conditionals(K_pa, K_ch, V, ilist)
        logQ_pVi_pai, sum_logQ_Vj_paj_pVi = self._get_conditionals(K_pa, K_ch, V_, ilist)

        # undirected part
        with torch.no_grad():
            sum_Vsk = - self.model.get_energy_neighbor(V, ilist)
            sum_Vsk_pVi = - self.model.get_energy_neighbor(V_, ilist)

        loss = logQ_Vi_pai - logQ_pVi_pai # batchsz
        loss += sum_logQ_Vj_paj - sum_logQ_Vj_paj_pVi # batchsz
        loss -= self.factor * (sum_Vsk - sum_Vsk_pVi) # batchsz
        loss = loss**2
        return loss.mean()

    def get_NLL(self, data, K_pa):
        nll = 0.
        B = self.batchsz_nll
        n_iters = data.shape[0] // B

        with torch.no_grad():
            for i in range(n_iters):
                V = data[i*B:(i+1)*B] # batchsz, vdim
                ilist = torch.arange(self.vdim).to(self.device) # vdim

                instance_idx, pa_idx = self.pgm.get_idx_from_K(K_pa, ilist)
                pa_idx = pa_idx.unsqueeze(0).tile([B, 1]).reshape(-1)
                added = self.vdim * torch.arange(B).to(self.device)
                instance_idx = (instance_idx.unsqueeze(0) + added.unsqueeze(1)).reshape(-1)

                V = V.unsqueeze(1).tile([1,self.vdim,1]).view(-1, self.vdim) # batchsz*vdim, vdim
                V_pa = torch.zeros_like(V)
                V_pa[instance_idx, pa_idx] = V[instance_idx, pa_idx]
                ilist = ilist.unsqueeze(0).tile([B, 1]).reshape(-1) # batchsz*vdim

                nll -= self.Qnet.log_prob(V_pa, ilist, V).mean().item()

            return nll / n_iters

    def sampleV(self, n_samp, temp=1, epsilon=0, uniform=False):
        if self.alg == "fixed":
            K_pa, top_order = self.pgm.K_pa_full, self.pgm.top_order_full
            top_order = top_order.repeat(n_samp, 1)
        else:
            K_pa, top_order = self.pgm.K_pa, self.pgm.top_order

        # sample from the joint distribution
        V = torch.zeros(n_samp, self.vdim).to(self.device)

        # Sample each V_i and its markov blanket from the associated sub-DAG
        for i in range(top_order.shape[1]):
            itself = top_order[:,i]
            instance_idx, pa_idx = self.pgm.get_idx_from_K(K_pa, itself)
            V_pa = torch.zeros_like(V)
            V_pa[instance_idx, pa_idx] = V[instance_idx, pa_idx]

            if self.alg == "fixed":
                probs = torch.sigmoid(self.Qnet(V_pa, itself))
                if epsilon > 0:
                    if uniform:
                        probs_off = 0.5 * torch.ones(n_samp).to(self.device)
                    else:
                        probs_off = torch.sigmoid(self.Qnet(V_pa, itself, temp=temp))
                    probs = (1 - epsilon) * probs + epsilon * probs_off
                v_samp = 2 * torch.bernoulli(probs) - 1  # batchsz
                V.scatter_(1, itself.view(-1, 1), v_samp.view(-1, 1))
            else:
                itself_mask = (itself >= 0)
                masked_itself = itself[itself_mask]
                probs = torch.sigmoid(self.Qnet(V_pa[itself_mask], masked_itself))
                if epsilon > 0:
                    if uniform:
                        probs_off = 0.5 * torch.ones(masked_itself.shape[0]).to(self.device)
                    else:
                        probs_off = torch.sigmoid(self.Qnet(V_pa[itself_mask], masked_itself, temp=temp))
                    probs = (1 - epsilon) * probs + epsilon * probs_off
                v_samp = 2 * torch.bernoulli(probs) - 1 # batchsz
                V[itself_mask] = V[itself_mask].scatter_(1, masked_itself.view(-1,1), v_samp.view(-1,1))

        return V