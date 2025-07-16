"""Sparsemax activation function.

Pytorch implementation of Sparsemax function from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

    def forward(self, input, dim=-1):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size

        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor

        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        device = input.device
        input = input.transpose(0, dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input

# @torch.no_grad()    
def sparsemax(input, dim=-1):
    device = input.device
    number_of_logits = input.size(dim)

    # Translate input by max for numerical stability
    # input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

    # Sort input in descending order.
    # (NOTE: Can be replaced with linear time selection method described here:
    # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
    zs = torch.sort(input=input, dim=dim, descending=True)[0]
    range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype)
    range = range.expand_as(zs)

    # Determine sparsity of projection
    bound = 1 + range * zs
    cumulative_sum_zs = torch.cumsum(zs, dim)
    is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
    k = torch.max(is_gt * range, dim, keepdim=True)[0]

    # Compute threshold function
    zs_sparse = is_gt * zs

    # Compute taus
    taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
    taus = taus.expand_as(input)

    # Sparsemax
    output = torch.max(torch.zeros_like(input), input - taus)
    return output


if __name__=="__main__":
    import numpy as np
    import matplotlib.pylab as pl
    import ot
    import ot.plot
    from ot.datasets import make_1D_gauss as gauss

    import torch
    from torch.autograd import Variable # 导入torch中Variable模块
        
    n = 100  # nb bins

    # bin positions
    # x = np.arange(n, dtype=np.float64)
    x = torch.arange(n, dtype=torch.float64)

    # Gaussian distributions
    # a = gauss(n, m=20, s=5)  # m= mean, s= std
    # b = gauss(n, m=60, s=10)
    a = torch.tensor(gauss(n, m=20, s=5))  # m= mean, s= std
    b = torch.tensor(gauss(n, m=60, s=10))

    # loss matrix
    M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
    M = M/M.max()

    def Euclidean_Bregman(r, c, M, lamb, iters=10, eps=1e-6):
        """
        Computes the optimal transport matrix and Slinkhorn distance using the
        Sinkhorn-Knopp algorithm
        """
        P = M / lamb

        for i in range(iters):
            P0 = P
            # Shape (n, )
            P = sparsemax(P, dim=-1)
            P = torch.diag(c)@P
            P = sparsemax(P, dim=-2)
            P = P@torch.diag(r)
            if (P0-P).max()<eps:break
        return P

    # Gs = ot.sinkhorn(a, b, M, lambd, verbose=True)
    lambd = 100
    OT_eb = Euclidean_Bregman(a, b, M, lambd)
    pl.figure(5, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, OT_eb, 'OT matrix Sinkhorn')