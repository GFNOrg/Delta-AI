import torch
import torch.nn as nn
from utils.distributions import BernoulliDist
from utils.pgm import get_idx_from_K

class Qnet(nn.Module):
    def __init__(self, vdim, xdim, hdim, dist, flow=False):
        super(Qnet, self).__init__()
        self.vdim = vdim
        self.xdim = xdim
        self.hdim = hdim

        self.flow = flow
        if self.flow:
            self.head_indim = self.vdim + 1
        else:
            self.head_indim = self.vdim

        self.outdim = 1

        self.marginals = nn.Parameter(torch.zeros(self.head_indim, self.outdim), requires_grad=True)
        self.l1 = nn.Sequential(nn.Linear(self.vdim, self.hdim), nn.LayerNorm(self.hdim), nn.ELU())
        self.l2 = nn.Sequential(nn.Linear(self.hdim, self.hdim), nn.LayerNorm(self.hdim), nn.ELU())
        self.l3 = nn.Sequential(nn.Linear(self.hdim, self.hdim), nn.LayerNorm(self.hdim), nn.ELU())
        self.W = nn.Parameter(0.01 * torch.randn(self.head_indim, self.hdim, self.outdim))
        self.b = nn.Parameter(torch.zeros(self.head_indim, 1, self.outdim))

        self._dist = dist

    def forward(self, V, ilist=None):
        out = self.l1(V)
        out = out + self.l2(out)
        out = out + self.l3(out)
        W = self.W[ilist]
        b = self.b[ilist]

        out = torch.bmm(out.unsqueeze(1), W) + b
        out = out.squeeze(1)
        out = torch.where((V.abs().sum(-1) == 0).unsqueeze(1), self.marginals[ilist], out)

        return out

    def logprob(self, V_pa, ilist, V):
        params = self.forward(V_pa, ilist)
        target = torch.gather(V, -1, ilist.unsqueeze(-1)).squeeze(-1) # batchsz
        return self._dist.logprob(params, target)

    def sample(self, V_pa, ilist, temp=1):
        params = self.forward(V_pa, ilist)
        return self._dist.sample(params, temp=temp)

    def get_flows(self, V):
        i_flows = self.vdim * torch.ones(V.shape[0], dtype=torch.long, device=V.device)
        return self.forward(V, i_flows)[:, 0]

class DeltaAI(nn.Module):
    def __init__(self, args, pgm, model_identity, objective="delta"):
        super(DeltaAI, self).__init__()

        self.device = args.device

        self.vdim = args.vdim
        self.xdim = args.xdim
        self.hdim = self.vdim - self.xdim

        self.model_identity = model_identity
        self.sampling_dag = args.sampling_dag

        self.batchsz = args.batchsz
        self.batchsz_proc = args.batchsz_proc
        self.batchsz_iw = args.batchsz_iw

        self._dist = BernoulliDist(self.vdim, reduction='none')

        self.pgm = pgm

        self.Qnet = Qnet(self.vdim, self.xdim, args.hdim_mae, self._dist,
                         flow=True if objective in ["tb", "db", "fldb", "mean_field"] else False)

        if objective == "delta":
            self.objective = self._delta_objective
        elif objective == "iw":
            self.objective = self._iw_objective
        elif objective == "tb":
            self.objective = self._tb_objective
        elif objective == "db":
            self.objective = self._db_objective
        elif objective == "fldb":
            self.objective = self._fldb_objective
        elif objective == "mean_field":
            self.objective = self._mean_field_objective
        elif objective in ["sleep", "none"]:
            pass
        else:
            raise NotImplementedError("objective {} not implemented".format(objective))

    def _get_conditionals(self, K_pa, K_ch, V, ilist):

        instance_pa_idx, pa_idx = get_idx_from_K(K_pa, ilist)
        instance_ch_idx, ch_idx = get_idx_from_K(K_ch, ilist)

        if K_pa.shape[0] > 1:
            K_pa_ch = torch.index_select(K_pa, 0, instance_ch_idx)

        V_pa = torch.zeros_like(V)

        if self.model_identity == "q":
            V_pa[:, :self.xdim] = V[:, :self.xdim]

        V_pa[instance_pa_idx, pa_idx] = V[instance_pa_idx, pa_idx]  # batchsz, vdim
        logQ_Vi_pai = self.Qnet.logprob(V_pa, ilist, V)

        # batch itself to avoid memory overflow
        num_chunks = (ch_idx.shape[0] + self.batchsz_proc - 1) // self.batchsz_proc
        sum_logQ_Vj_paj = torch.zeros_like(V)
        for i in range(num_chunks):
            S = i * self.batchsz_proc
            E = min((i + 1) * self.batchsz_proc, ch_idx.shape[0])
            itself_batch = ch_idx[S:E]
            instance_ch_idx_batch = instance_ch_idx[S:E]

            if K_pa.shape[0] > 1:
                K_pa_batch = K_pa_ch[S:E]
            else:
                K_pa_batch = K_pa

            instance_idx, pa_idx = get_idx_from_K(K_pa_batch, itself_batch)

            V_pa = torch.zeros(itself_batch.shape[0], self.vdim, device=self.device)
            V_ch_i = torch.index_select(V, 0, instance_ch_idx_batch)
            V_pa_j = torch.index_select(V_ch_i, 0, instance_idx)
            V_pa[instance_idx, pa_idx] = torch.gather(V_pa_j, 1, pa_idx.unsqueeze(1)).squeeze(1)

            if self.model_identity == "q":
                V_pa[:, :self.xdim] = V_ch_i[:, :self.xdim]

            ll = self.Qnet.logprob(V_pa, itself_batch, V_ch_i)
            sum_logQ_Vj_paj[instance_ch_idx_batch, itself_batch] = ll

        sum_logQ_Vj_paj = sum_logQ_Vj_paj.sum(1)

        return logQ_Vi_pai, sum_logQ_Vj_paj

    def _delta_objective(self, p, V, ilist):

        K_pa = self.pgm.K_pa[self.sampling_dag][self.model_identity]
        K_ch = self.pgm.K_ch[self.sampling_dag][self.model_identity]

        logQ_Vi_pai, sum_logQ_Vj_paj = self._get_conditionals(K_pa, K_ch, V, ilist)

        V_ = self._dist.perturb(V, ilist)
        logQ_pVi_pai, sum_logQ_Vj_paj_pVi = self._get_conditionals(K_pa, K_ch, V_, ilist)

        K_pa = self.pgm.K_pa[self.sampling_dag]["p"]
        K_ch = self.pgm.K_ch[self.sampling_dag]["p"]
        with torch.no_grad():
            logP_Vi_pai, sum_logP_Vj_paj = p._get_conditionals(K_pa, K_ch, V, ilist)
            logP_pVi_pai, sum_logP_Vj_paj_pVi = p._get_conditionals(K_pa, K_ch, V_, ilist)

        loss = (logQ_Vi_pai - logQ_pVi_pai) + (sum_logQ_Vj_paj - sum_logQ_Vj_paj_pVi)
        loss -= (logP_Vi_pai - logP_pVi_pai) + (sum_logP_Vj_paj - sum_logP_Vj_paj_pVi)

        return (loss**2).mean()

    def _tb_objective(self, p, V, ilist):
            logPF, logZcond = self.probV(V, "full", log=True, return_flow="Z")

            with torch.no_grad():
                logR = p.probV(V, "full", log=True)

            loss = (logZcond.squeeze(1) + logPF - logR) ** 2

            return loss.mean()

    def _db_objective(self, p, V, ilist):

        logPF, logF = self.probV(V, "full", log=True, return_flow="all", reduction="none")

        with torch.no_grad():
            logR = p.probV(V, "full", log=True, reduction="sum")
            logF[:,-1] = logR

        loss = (logF[:, :-1] + logPF - logF[:, 1:])**2
        return loss.mean()

    def _fldb_objective(self, p, V, ilist):
        logPF, logF = self.probV(V, "full", log=True, return_flow="all", reduction="none", fl_source=p)

        with torch.no_grad():
            logR = p.probV(V, "full", log=True, reduction="sum")
            logF[:, -1] = logR

        loss = (logF[:, :-1] + logPF - logF[:, 1:]) ** 2
        return loss.mean()

    def _iw_objective(self, p, V, ilist):
        logprobQF = self.probV(V, "full", log=True, repeat_dags=self.batchsz_iw).view(-1, self.batchsz_iw)
        with torch.no_grad():
            logprobQB = p.probV(V, "full", log=True, repeat_dags=self.batchsz_iw).view(-1, self.batchsz_iw)
            w = (logprobQB - logprobQF).softmax(1)
        loss = (w * (- logprobQF)).sum(1)

        return loss.mean()

    def sampleV(self, batchsz, sampling_dag, temp=1, epsilon=0, X=None, H=None,
                repeat_dags=1):

        with torch.no_grad():
            V = torch.zeros(batchsz, self.vdim).to(self.device)
            K_pa = self.pgm.K_pa[sampling_dag][self.model_identity]
            top_order = self.pgm.top_order[sampling_dag][self.model_identity]

            if X is not None:
                V[:, :self.xdim] = X

            if sampling_dag == "full":
                if H is not None:
                    V[:, self.xdim:] = H
                    # Remove all H from top_order
                    top_order = top_order[:, (top_order < self.xdim).squeeze(0)]

                top_order = top_order.repeat(batchsz, 1)
            else:
                if H is not None:
                    raise NotImplementedError
                if repeat_dags > 1:
                    top_order = torch.repeat_interleave(top_order, repeat_dags, dim=0)
                    K_pa = torch.repeat_interleave(K_pa, repeat_dags, dim=0)

            V_pa_base = torch.zeros_like(V)
            if self.model_identity == "q":
                V_pa_base[:, :self.xdim] = V[:, :self.xdim]
            V_pa = V_pa_base.clone()

            # Sample each V_i and its markov blanket from the sub-DAG
            for i in range(top_order.shape[1]):
                itself = top_order[:,i]

                instance_idx, pa_idx = get_idx_from_K(K_pa, itself)
                V_pa[instance_idx, pa_idx] = V[instance_idx, pa_idx]

                if sampling_dag == "full":
                    v_samp = self.Qnet.sample(V_pa, itself)

                    if epsilon > 0:
                        v_samp_off = self.Qnet.sample(V_pa, itself, temp=temp)

                        off_mask = torch.rand_like(v_samp) < epsilon
                        v_samp = ~off_mask * v_samp + off_mask * v_samp_off

                    V.scatter_(1, itself.view(-1, 1), v_samp.view(-1, 1))

                else:
                    itself_mask = (itself >= 0)
                    masked_itself = itself[itself_mask]

                    v_samp = self.Qnet.sample(V_pa[itself_mask], masked_itself)

                    if epsilon > 0:
                        v_samp_off = self.Qnet.sample(V_pa[itself_mask], masked_itself, temp=temp)

                        off_mask = torch.rand_like(v_samp) < epsilon
                        v_samp = ~off_mask * v_samp + off_mask * v_samp_off

                    V[itself_mask] = V[itself_mask].scatter_(1, masked_itself.view(-1,1), v_samp.view(-1,1))

                # Reset V_pa
                V_pa.copy_(V_pa_base)
        return V

    def probV(self, V, sampling_dag, include_H=True,
              log=False, return_flow=None, reduction="sum", expected_value=True,
              top_order=None, fl_source=None, repeat_dags=1):

        K_pa = self.pgm.K_pa[sampling_dag][self.model_identity]
        if top_order is None:
            top_order = self.pgm.top_order[sampling_dag][self.model_identity]

        if sampling_dag == "full":
            if not include_H:
                top_order = top_order[:, (top_order < self.xdim).squeeze(0)]
            top_order = top_order.repeat(V.shape[0], 1)
        else:
            if repeat_dags > 1:
                top_order = torch.repeat_interleave(top_order, repeat_dags, dim=0)
                K_pa = torch.repeat_interleave(K_pa, repeat_dags, dim=0)

        n_steps = top_order.shape[1]

        if reduction == "sum":
            probs = torch.zeros((V.shape[0],)).to(self.device)
        elif reduction == "none":
            probs = torch.zeros((V.shape[0], n_steps)).to(self.device)
        else:
            raise ValueError

        V_pa_base = torch.zeros_like(V)
        if self.model_identity == "q":
            V_pa_base[:, :self.xdim] = V[:, :self.xdim]
        V_pa = V_pa_base.clone()

        if return_flow:
            V_state = V_pa.clone()
            if return_flow == "all":
                logF = torch.zeros((V.shape[0], n_steps + 1)).to(self.device)
            elif return_flow == "Z":
                logF = torch.zeros((V.shape[0], 1)).to(self.device)
                logF[:, 0] = self.Qnet.get_flows(V_state)
            else:
                raise ValueError

        for i in range(top_order.shape[1]):

            itself = top_order[:, i]
            i_step = i * torch.ones_like(itself).view(-1, 1)

            if return_flow == "all":
                log_state_flow = self.Qnet.get_flows(V_state)
                if fl_source is not None:
                    with torch.no_grad():
                        log_state_flow += fl_source.probV(V_state, sampling_dag, log=True, reduction="sum")
                logF.scatter_(1, i_step, log_state_flow.view(-1,1))
                V_state = torch.scatter(V_state, 1, itself.view(-1, 1), V[torch.arange(V.shape[0]), itself].view(-1, 1))

            instance_idx, pa_idx = get_idx_from_K(K_pa, itself)
            V_pa[instance_idx, pa_idx] = V[instance_idx, pa_idx]

            if sampling_dag == "full":
                if log:
                    p = self.Qnet.logprob(V_pa, itself, V)
                else:
                    if expected_value:
                        p = torch.sigmoid(self.Qnet(V_pa, itself)[:, 0])
                    else:
                        raise NotImplementedError

                if reduction == "sum":
                    probs += p
                elif reduction == "none":
                    probs.scatter_(1, i_step, p.view(-1, 1))
                else:
                    raise NotImplementedError

            else:
                itself_mask = (itself >= 0)
                masked_itself = itself[itself_mask]

                if log:
                    p = self.Qnet.logprob(V_pa[itself_mask], masked_itself, V[itself_mask])
                else:
                    p = torch.sigmoid(self.Qnet(V_pa[itself_mask], masked_itself)[:, 0])

                if reduction == "sum":
                    probs[itself_mask] += p
                elif reduction == "none":
                    probs[itself_mask] = probs[itself_mask].scatter_(1, masked_itself.view(-1, 1), p.view(-1, 1))
                else:
                    raise NotImplementedError

            # Reset V_pa
            V_pa = V_pa_base.clone()

        if return_flow:
            return probs, logF
        else:
            return probs

    def get_flip_prob(self, V, i):
        # Accessory method for compatibility with gibbs sampler
        cond_list = self.pgm.H_cond[i-self.xdim] # Nodes with conditionals involving node i
        top_order = torch.Tensor([cond_list]).long().to(V.device)
        V_pos, V_neg = V.clone(), V.clone()
        V_pos[:, i], V_neg[:, i] = 1., -1.
        logprob_pos = self.probV(V_pos, "full", include_H=True, reduction="sum", log=True, top_order=top_order)
        logprob_neg = self.probV(V_neg, "full", include_H=True, reduction="sum", log=True, top_order=top_order)

        return torch.sigmoid(logprob_pos-logprob_neg)
