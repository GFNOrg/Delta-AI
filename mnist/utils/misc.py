import os
import torch
import numpy as np

def makedir(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")

def get_NLL_importance(q, p, X, batchsz_importance):

    batch_size = X.shape[0] * batchsz_importance
    X_repeat = torch.repeat_interleave(X.unsqueeze(1), batchsz_importance, dim=1).view(batch_size, -1)
    V_F = q.sampleV(batch_size, "full", temp=1, epsilon=0, X=X_repeat)
    #  logsumexp ( log QB(Hi,X) - log QF(Hi|X) ] - log K
    logprob_QB = p.probV(V_F, "full", log=True, reduction="sum").view(-1, batchsz_importance)
    logprob_QF = q.probV(V_F, "full", log=True, reduction="sum").view(-1, batchsz_importance)
    ll = torch.logsumexp(logprob_QB - logprob_QF, dim=1) - np.log(batchsz_importance)

    return -ll
