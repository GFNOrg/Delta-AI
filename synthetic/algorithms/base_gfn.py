import torch
import torch.nn as nn

class Qnet(nn.Module):
    def __init__(self, vdim, hdim, odim):
        super(Qnet, self).__init__()
        self.vdim = vdim
        self.hdim = hdim
        self.odim = odim

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
            nn.Linear(self.hdim, self.odim)
        )

    def forward(self, V):
        return self.layers(V)

class GFlowNet(nn.Module):
    def __init__(self, args, device, model):
        super(GFlowNet, self).__init__()

        self.device = device
        self.vdim = args.vdim
        self.hdim = args.hdim
        self.batchsz = args.batchsz

        self.alg = args.alg
        self.uniform_pb = args.uniform_pb
        self.temp = args.temp
        self.rand_coef = args.rand_coef
        self.back_ratio = args.back_ratio
        self.lamda = args.lamda

        self.scorer = lambda V: - model.get_energy(V)
        self.partial_scorer = (lambda V: - model.get_energy(V)) \
                if args.forward_looking else None

        self.Qnet1 = Qnet(self.vdim, self.hdim, 3*self.vdim)
        self.Qnet2 = Qnet(self.vdim, self.hdim, 1)
        self.logZ = nn.Parameter(torch.tensor(0.))

    def _get_parts(self, V):
        logits = self.Qnet1(V)
        add_logits = logits[:, :2*self.vdim]
        del_logits = logits[:, 2*self.vdim:]
        logF = self.Qnet2(V).squeeze(-1)

        if self.uniform_pb:
            correction = add_logits.reshape(-1, 2).logsumexp(-1)
            correction = correction.unsqueeze(-1).repeat(1, 2).view(add_logits.shape)
            add_logits = add_logits - correction
            del_logits = del_logits * 0

        return add_logits, del_logits, logF

    def _get_loss_fn(self, logF, logPF, logPB):
        if self.alg == 'tb':
            return (logF[:,0] + logPF.sum(1) - logF[:,-1] - logPB.sum(1))**2
        elif self.alg == 'db':
            return ((logF[:, :-1] + logPF - logF[:, 1:] - logPB)**2).mean(1)
        elif self.alg == 'subtb':
            loss, wsum = 0., 0.
            for i in range(self.vdim):
                logPF_cumsum = logPF[:,i:].cumsum(1) # batchsz, xdim-i
                logPB_cumsum = logPB[:,i:].cumsum(1) # batchsz, xdim-i
                logF_diff = logF[:,i].unsqueeze(1) - logF[:,i+1:] # batchsz, xdim-i
                w = torch.pow(self.lamda, 1 + torch.arange(self.vdim - i)).to(self.device)
                l = w * (logPF_cumsum - logPB_cumsum + logF_diff)**2
                wsum += w.sum().item()
                loss = loss + l.sum(1)
            return loss / wsum
        else:
            raise NotImplementedError

    def get_loss(self, batch, is_eval=False):

        INF = 1e+20
        EPS = 1e-20
        back_ratio = 1. if is_eval else self.back_ratio

        if back_ratio < 1.:
            V = torch.zeros((self.batchsz, self.vdim)).to(self.device)
            log_pb = torch.zeros((self.batchsz, self.vdim)).to(self.device)
            log_pf = torch.zeros_like(log_pb)
            log_flow = torch.zeros((self.batchsz, self.vdim + 1)).to(self.device)

            if self.alg == "tb":
                log_flow[:,0] = self.logZ
            else:
                V0 = torch.zeros_like(V)
                _, _, log_flow[:,0] = self._get_parts(V0)

            for step in range(self.vdim + 1):
                add_logits, del_logits, logF = self._get_parts(V)

                if self.partial_scorer is not None:
                    logF += self.partial_scorer(V)

                if step > 0:
                    mask = (V == 0).float()
                    log_pb[:, step - 1] = \
                            (del_logits - INF * mask).log_softmax(1).gather(1, add_locs).squeeze(1)
                    log_flow[:, step] = logF

                if step < self.vdim:
                    mask = (V != 0).unsqueeze(2).repeat(1, 1, 2).reshape(self.batchsz, 2*self.vdim).float()
                    add_logits = (add_logits - INF * mask).float()
                    add_prob = add_logits.softmax(1)

                    add_prob = add_prob ** (1 / self.temp)
                    add_prob = add_prob / (EPS + add_prob.sum(1, keepdim=True))
                    add_prob = (1 - self.rand_coef) * add_prob + \
                               self.rand_coef * (1 - mask) / (EPS + (1 - mask).sum(1)).unsqueeze(1)
                    add_sample = add_prob.multinomial(1) # batchsz, 1

                    add_locs, add_values = add_sample // 2, 2 * (add_sample % 2) - 1
                    log_pf[:, step] = add_logits.log_softmax(1).gather(1, add_sample).squeeze(1)
                    V = V.scatter(1, add_locs, add_values.float())

            assert torch.all(V != 0)

            log_flow[:,-1] = self.scorer(V)
            loss_forth = self._get_loss_fn(log_flow, log_pf, log_pb)
        else:
            loss_forth = torch.tensor(0.).to(self.device)

        loss_mle = torch.zeros(self.batchsz, self.vdim).to(self.device)

        if back_ratio <= 0.:
            data_log_pb = torch.zeros(self.batchsz, self.vdim).to(self.device)
            loss_back = torch.tensor(0.).to(self.device)
        else:
            assert batch is not None
            assert batch.shape[0] == self.batchsz
            V = batch
            data_log_pb = torch.zeros(self.batchsz, self.vdim).to(self.device)
            log_flow = torch.zeros(self.batchsz, self.vdim + 1).to(self.device)

            if self.alg == "tb":
                log_flow[:,-1] = self.logZ
            else:
                V0 = torch.zeros_like(V)
                _, _, log_flow[:,-1] = self._get_parts(V0)

            for step in range(self.vdim + 1):
                add_logits, del_logits, logF = self._get_parts(V)

                if self.partial_scorer is not None:
                    logF += self.partial_scorer(V)

                if step > 0:
                    mask = (V != 0).unsqueeze(2).repeat(1, 1, 2)
                    mask = mask.reshape(self.batchsz, 2 * self.vdim).float()

                    add_sample = 2 * del_locs + (deleted_values == 1).long()
                    add_logits = (add_logits - INF * mask).float()
                    loss_mle[:, step - 1] = add_logits.log_softmax(1).gather(1, add_sample).squeeze(1)

                if step < self.vdim:
                    mask = (V == 0).float()
                    del_logits = (del_logits - INF * mask).float()
                    del_prob = del_logits.softmax(1)
                    if not is_eval:
                        del_prob = (1 - self.rand_coef) * del_prob + self.rand_coef * \
                                (1 - mask) / (EPS + (1 - mask).sum(1)).unsqueeze(1)
                    del_locs = del_prob.multinomial(1)  # row sum not need to be 1
                    deleted_values = V.gather(1, del_locs)
                    data_log_pb[:, step] = del_logits.log_softmax(1).gather(1, del_locs).squeeze(1)
                    log_flow[:, step] = logF
                    del_values = torch.zeros(self.batchsz, 1).to(self.device)
                    V = V.scatter(1, del_locs, del_values)

            log_flow[:, 0] = self.scorer(batch).detach()
            loss_back = self._get_loss_fn(log_flow, data_log_pb, loss_mle)

        loss_gfn = (1 - back_ratio) * loss_forth + back_ratio * loss_back
        loss_mle = - loss_mle.sum(1)

        if is_eval:
            return loss_mle, data_log_pb.sum(1)
        else:
            return loss_gfn.mean()

    def get_NLL(self, data, mc_size):
        with torch.no_grad():
            logp_ls = []
            for _ in range(mc_size):
                loss_mle_list, data_log_pb_list = [], []
                N = data.shape[0]
                assert N % self.batchsz == 0

                for i in range(N // self.batchsz):
                    batch = data[i*self.batchsz:(i+1)*self.batchsz]
                    loss_mle, data_log_pb = self.get_loss(batch, is_eval=True)
                    loss_mle_list.append(loss_mle)
                    data_log_pb_list.append(data_log_pb)

                loss_mle = torch.cat(loss_mle_list, 0)
                data_log_pb = torch.cat(data_log_pb_list, 0)
                logpj = - loss_mle.detach().cpu() - data_log_pb.detach().cpu()
                logp_ls.append(logpj.reshape(logpj.shape[0], -1))

            batch_logp = torch.logsumexp(torch.cat(logp_ls, dim=1), dim=1)  # (bs,)
            ll = batch_logp.mean() - torch.tensor(mc_size).log()
            return -ll / self.vdim
