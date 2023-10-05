import torch

def get_gibbs_from_data(model, V, K):
    return gibbs_sampling_batch(model, V, K)

def get_gibbs_sample(model, n, K=10000, KA=0):
    p = 0.5*torch.ones(n, model.vdim).cuda()
    V = 2 * torch.bernoulli(p) - 1 # n, vdim
    return gibbs_sampling_batch(model, V, K, KA)

def gibbs_sampling_batch(model, V, K, KA):
    # V: batchsz, vdim
    V = V.clone()

    # annealing
    for k in range(KA):
        for i in range(model.vdim):
            invtemp = k / KA
            p = model.get_conditional(V, i, invtemp) # bs
            V[:,i].data.copy_(2 * torch.bernoulli(p) - 1) # bs

    # start
    for k in range(K):
        for i in range(model.vdim):
            p = model.get_conditional(V, i) # bs
            V[:,i].data.copy_(2 * torch.bernoulli(p) - 1) # bs
    return V # batchsz, vdim
