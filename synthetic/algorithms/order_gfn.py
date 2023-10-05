import torch
import torch.nn as nn

class Qnet(nn.Module):
    def __init__(self, vdim, hdim):
        super(Qnet, self).__init__()
        self.vdim = vdim
        self.hdim = hdim

        self.layers = nn.Sequential(
            nn.Linear(self.vdim, self.hdim),
            nn.LayerNorm(self.hdim),
            nn.ReLU(),
            nn.Linear(self.hdim, self.hdim),
            nn.LayerNorm(self.hdim),
            nn.ReLU(),
            nn.Linear(self.hdim, self.hdim),
            nn.LayerNorm(self.hdim),
            nn.ReLU(),
        )
        self.W = nn.Parameter(0.01*torch.randn(self.vdim, self.hdim, 1))
        self.b = nn.Parameter(torch.zeros(self.vdim, 1, 1))
        self.marginals = nn.Parameter(torch.zeros(self.vdim))

    def forward(self, V, ilist, temp=1):
        # V: batchsz, vdim
        x = self.layers(V).unsqueeze(1) # batchsz, 1, hdim
        W = self.W[ilist] # batchsz, hdim, 1
        b = self.b[ilist] # batchsz, 1, 1
        x = (x.bmm(W) + b).squeeze() # batchsz

        out = torch.where(V.abs().sum(1) == 0,
                          torch.sigmoid(self.marginals[ilist] / temp),
                          torch.sigmoid(x / temp))
        return out

class Fnet(nn.Module):
    def __init__(self, vdim, hdim):
        super(Fnet, self).__init__()
        self.vdim = vdim
        self.hdim = hdim

        self.layers = nn.Sequential(
            nn.Linear(self.vdim, self.hdim),
            nn.LayerNorm(self.hdim),
            nn.ReLU(),
            nn.Linear(self.hdim, self.hdim),
            nn.LayerNorm(self.hdim),
            nn.ReLU(),
            nn.Linear(self.hdim, self.hdim),
            nn.LayerNorm(self.hdim),
            nn.ReLU(),
            nn.Linear(self.hdim, 1),
        )

    def forward(self, V):
        # V: batchsz, vdim
        x = self.layers(V) # batchsz, 1
        return x.squeeze(-1)

class GFlowNet(nn.Module):
    def __init__(self, args, device, model, pgm):
        super(GFlowNet, self).__init__()

        self.vdim = args.vdim
        self.hdim = args.hdim
        self.batchsz = args.batchsz
        self.batchsz_nll = args.batchsz_nll
        self.device = device

        self.model = model
        self.pgm = pgm

        self.alg = args.alg
        self.Qnet = Qnet(self.vdim, self.hdim)
        self.Fnet = Fnet(self.vdim, self.hdim)
        self.logZ = nn.Parameter(torch.tensor(0.))

        self.bce_loss = nn.BCELoss(reduction='none')

        self.scorer = lambda V: - model.get_energy(V).detach()
        self.partial_scorer = (lambda V: - model.get_energy(V).detach()) \
                if "fl" in args.alg else None

    def sampleV(self, temp=1, epsilon=0, uniform=False, data=None):
        K_pa, top_order = self.pgm.K_pa_full, self.pgm.top_order_full
        top_order = top_order.repeat(self.batchsz, 1)

        # sample from the joint Q distribution
        if data is None:
            V = torch.zeros(self.batchsz, self.vdim).to(self.device)
        else:
            V = data

        log_pf = torch.zeros((self.batchsz, self.vdim)).to(self.device)
        log_flow = torch.zeros((self.batchsz, self.vdim + 1)).to(self.device)

        if self.alg == "tb":
            log_flow[:,0] = self.logZ
        else:
            V0 = torch.zeros_like(V)
            log_flow[:,0] = self.Fnet(V0)

        # Sample each V_i and its markov blanket from the associated marginalized sub-DAG
        for i in range(top_order.shape[1]):
            itself = top_order[:,i]

            instance_idx, pa_idx = self.pgm.get_idx_from_K(K_pa, itself)
            V_pa = torch.zeros_like(V)
            V_pa[instance_idx, pa_idx] = V[instance_idx, pa_idx]

            probs_on = self.Qnet(V_pa, itself)

            if data is not None:
                v_samp = V.gather(1, itself.unsqueeze(1)).squeeze(1)
                samp = (v_samp == 1)
            else:
                if epsilon > 0:
                    if uniform:
                        probs_off = 0.5 * torch.ones(self.batchsz).to(self.device)
                    else:
                        probs_off = self.Qnet(V_pa, itself, temp=temp)

                    probs = (1 - epsilon) * probs_on + epsilon * probs_off
                else:
                    probs = probs_on

                samp = torch.bernoulli(probs)
                v_samp = 2 * samp - 1  # batchsz
                V = V.scatter_(1, itself.view(-1, 1), v_samp.view(-1, 1))
                samp = samp.bool()

            # PF
            probs_on[~samp] = (1 - probs_on[~samp])
            log_pf[:,i] = probs_on.log()

            # F
            logF = self.Fnet(V.clone().detach())
            if self.partial_scorer is not None:
                logF += self.partial_scorer(V.clone().detach())
            log_flow[:,i+1] = logF

        # R
        log_flow[:,-1] = self.scorer(V)

        if self.alg == "tb":
            loss = (log_flow[:,0] + log_pf.sum(1) - log_flow[:,-1])**2
        elif self.alg in ["db", "dbfl"]:
            loss = ((log_flow[:, :-1] + log_pf - log_flow[:, 1:])**2).mean(1)
        else:
            raise NotImplementedError

        return V, loss.mean()

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

                probs = self.Qnet(V_pa, ilist) # batchsz*vdim
                log_probs = -self.bce_loss(probs,
                        torch.gather((V+1)/2, -1, ilist.unsqueeze(-1)).squeeze(-1)) # batchsz*vdim
                nll -= log_probs.mean().item()

            return nll / n_iters
