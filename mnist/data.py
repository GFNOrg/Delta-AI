import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def load_mnist(batch_size_train=1000, batch_size_test=1000, n_iters=50000, path=""):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: x.view(28*28)),
         transforms.Lambda(lambda x: 2*torch.bernoulli(x) -1)
         ]
    )

    root_dir = os.path.join(path, "data")
    train_set = torchvision.datasets.MNIST(root=root_dir, train=True,
                                           download=True, transform=transform)
    train_sampler = torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=batch_size_train*n_iters)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, sampler=train_sampler)

    test_set = torchvision.datasets.MNIST(root=root_dir, train=False,
                                         download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)

    test_set = torch.stack([test_set[i][0] for i in range(len(test_set))])

    return train_loader, test_loader, test_set

class Pyramid:
    def __init__(self, args):
        self.K = args.K
        self.stride = args.stride
        self.h_depth = args.h_depth
        self.device = args.device

        self.width = None
        self.height = None
        self.layer_start = None

        self._set_pyramid_dims()
        self._set_adj()

    def get_adj(self):
        return self.adj

    def get_dims(self):
        return self.vdim, self.xdim, self.width, self.height

    def _set_pyramid_dims(self):

        width = [28]
        height = [28]

        for d, (k, stride) in enumerate(zip(self.K, self.stride)):

            dim_dict = {}

            dim_dict['width'] = width[d]
            if dim_dict['width'] == 1 and k > 1:
                raise AssertionError(f"width at layer {d} is already 1, cannot reduce further")

            dim_dict['height'] = height[d]
            if dim_dict['height'] == 1 and k > 1:
                raise AssertionError(f"height at layer {d} is already 1, cannot reduce further")

            for key, val in dim_dict.items():
                if k == 1:
                    if key == "width":
                        width.append(width[d])
                    else:
                        height.append(height[d])
                else:
                    if stride < 1:
                        raise AssertionError(f"Stride must be >=1")
                    if val < k:
                        raise AssertionError(f"K={k} bigger than {key}={val} at layer {d}")
                    if (val - k) % stride != 0:
                        raise AssertionError(f"K={k} not compatible with stride {stride} at layer {d} with {key}={val}")

                    upper = (val - k) // stride + 1

                    if key == 'width':
                        width.append(upper)
                    else:
                        height.append(upper)

        vdim_list = [w * h * self.h_depth if i > 0 else w * h for i, (w, h) in enumerate(zip(width, height))]
        cumsum_vdim = np.cumsum(vdim_list)
        layer_start = [0] + list(cumsum_vdim[:-1])

        vdim = sum(vdim_list)
        if len(layer_start) == 1:
            xdim = vdim
        else:
            xdim = layer_start[1]

        self.vdim = vdim
        self.xdim = xdim
        self.width = width
        self.height = height
        self.layer_start = layer_start

    def _set_adj(self):

        K_w = self.K
        stride_w = self.stride
        K_h = self.K
        stride_h = self.stride

        adj = torch.zeros(self.vdim, self.vdim).to(self.device)

        for d in range(1, len(self.K) + 1):
            w = self.width[d]
            h = self.height[d]
            if len(self.layer_start) == 1:
                i_start = 0
            else:
                i_start = self.layer_start[d]
            k_w = K_w[d - 1]
            k_h = K_h[d - 1]
            s_w = stride_w[d - 1]
            s_h = stride_h[d - 1]
            w_prev = self.width[d - 1]
            h_prev = self.height[d - 1]
            i_start_prev = self.layer_start[d - 1]

            for i in range(h):
                for j in range(w):

                    if k_w == 1 and k_h == 1:
                        args_list = [i * w_prev + j + i_start_prev]
                    else:
                        args_list = [((i * s_h) + i_prev) * w_prev + (j * s_w) + j_prev + i_start_prev
                                     for i_prev in range(k_h) for j_prev in range(k_w)]

                    if d > 1:
                        # layer below has depth h_depth
                        args_list = [a + (hd_prev * w_prev * h_prev) for hd_prev in range(self.h_depth) for a in
                                     args_list]

                    # if not (len(self.last_factor_dims) != 0 and d == len(self.width)-1):

                    source = i * w + j + i_start
                    source_depth = [source + (hd * w * h) for hd in range(self.h_depth)]
                    args_list += source_depth

                    args_list = torch.Tensor(args_list).to(dtype=torch.int64, device=adj.device)
                    adj[args_list, args_list.view(-1, 1)] = 1.

        # Remove edges in X
        adj[:self.xdim, :self.xdim] = 0.

        # Remove diagonal
        adj = adj - torch.diag(torch.diag(adj))

        self.adj = adj