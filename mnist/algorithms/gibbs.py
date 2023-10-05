import torch
import numpy as np

class GibbsSampler:
    def __init__(self, model, K):
        self.model = model
        self.K = K
    def get_gibbs_from_data(self, V, start_idx=0):
        batch = self.gibbs_sampling_batch(V, start_idx=start_idx)

        return batch

    def gibbs_sampling_batch(self, V, start_idx=0):
        V = V.clone()
        for k in range(self.K):
            for i in np.random.permutation(np.arange(start_idx, self.model.vdim)):
                p = self.model.get_flip_prob(V, i)
                V[:,i].data.copy_(2 * torch.bernoulli(p) - 1)
        return V