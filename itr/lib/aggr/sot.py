""" Sparse Optimal Transport """
import ot
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.aggr.pooling import AvePool
from lib.modules.utils import get_mask,get_fgsims,get_fgmask
import torch.multiprocessing as mp

# fine-grained sinkhorn distance
class SparseOT(nn.Module):
    def __init__(self, iters=3, lamb=5e-2, split=1,):
        super(SparseOT, self).__init__()
        self.eps = 1e-6
        self.iters = iters
        self.lamb = lamb
        self.split = split

    def _ave_init(self, lens):
        b = lens.shape[0]
        max_l = int(lens.max())

        mask = torch.arange(max_l).expand(b, max_l).to(lens.device)
        mask = (mask<lens.long().unsqueeze(dim=1)).float().to(lens.device)
        mask = mask*(1/mask.sum(dim=-1, keepdim=True))
        return mask

    def sparsemax(self, input, margin, dim=-1):
        """
            return the threshold
        """
        device = input.device
        adim = -3-dim
        n = input.size(dim)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=n + 1, step=1, device=device).type(input.type()).unsqueeze(adim)

        # Determine sparsity of projection
        bound = margin.unsqueeze(dim) + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        # is_gt = bound<=cumulative_sum_zs
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim)[0]
        k[k<1] = 1

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim) - margin) / k
        # taus = taus.expand_as(input)

        # Sparsemax
        return taus
    
    def emd(self, sims, r, c):
        M = 1-sims
        p = ot.emd(r, c, M)
        return (p*sims).sum()

    @torch.no_grad()
    def Euclidean_Bregman(self, M, r, c):
        """
        Computes the optimal transport matrix and Slinkhorn distance using the
        Sinkhorn-Knopp algorithm
        """
        alpha = torch.zeros_like(r)
        for i in range(self.iters):
            # update beta
            omega = alpha.unsqueeze(dim=-1)-M
            beta = -self.sparsemax(omega, self.lamb*c, dim=-2)
            # update alpha
            omega = beta.unsqueeze(dim=-2)-M
            alpha = -self.sparsemax(omega, self.lamb*r, dim=-1)

        P = (beta.unsqueeze(dim=-2)+alpha.unsqueeze(dim=-1)-M)/self.lamb
        P[P<0]=0
        return P

    @torch.no_grad()
    def Bregman(self, sims, r, c):
        """
        Computes the optimal transport matrix and Slinkhorn distance using the
        Bregman algorithm
        """
        P = torch.exp(sims / self.lamb)    
        # Avoiding poor math condition
        P = P / (P.sum(dim=[-2,-1], keepdim=True)+self.eps)

        for i in range(self.iters):
            # Shape (n, )
            # P0 = P
            u = P.sum(dim=[-1],) + self.eps # u(0)
            P = P * (r / u).unsqueeze(dim=-1) # u(0)*
            v = P.sum(dim=[-2]) + self.eps
            P = P * (c / v).unsqueeze(dim=-2)
        return P

    def _mask(self, sims, topk=1):
        pass

    # def forward(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
    #     bi, bt = imgs.size(0), caps.size(0)
    #     max_r,max_w = int(img_lens.max()),int(cap_lens.max())
    #     fg_sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
    #     mask = get_fgmask(img_lens,cap_lens)

    #     sims = torch.zeros(bi, bt).to(device=caps.device)
    #     step = bi//self.split
    #     for i in range(self.split):
    #         beg = step*i
    #         ed = bi if i+1==self.split else step*(i+1) 
    #         r,c = self._ave_init(mask[beg:ed])
    #         tp = self.Euclidean_Bregman((1-fg_sims[beg:ed]).masked_fill(mask[beg:ed] == 0, 2), r, c)
    #         sims[beg:ed] = (fg_sims[beg:ed]*tp*mask[beg:ed]).sum(dim=[-2,-1])
    #     else: return sims

    ## 
    # def forward(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
    #     bi, bt = imgs.size(0), caps.size(0)
    #     max_r,max_w = int(img_lens.max()),int(cap_lens.max())
    #     imgs = imgs[:,:max_r]
    #     caps = caps[:,:max_w]
    #     fg_sims = torch.einsum("ikd,tld->iktl", imgs, caps)
    #     fg_sims = fg_sims.reshape(bi*max_r, bt*max_w)

    #     # mask = get_fgmask(img_lens,cap_lens)
    #     # mask = mask.permute(0,2,1,3).reshape(bi*max_r, bt*max_w)

    #     r = self._ave_init(img_lens)
    #     r = r.reshape(bi*max_r)
    #     c = self._ave_init(cap_lens)
    #     c = c.reshape(bt*max_w)
    #     ## 

    #     P = ot.emd(r,c,1-fg_sims)
    #     sims = (P*fg_sims).reshape(bi, max_r, bt, max_w).sum(dim=[-3,-1])
    #     return sims

    def forward(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r]
        caps = caps[:,:max_w]

        fg_sims = get_fgsims(imgs, caps)
        r = self._ave_init(img_lens)
        r = r.unsqueeze(dim=1).repeat(1,fg_sims.size(1),1)
        c = self._ave_init(cap_lens)
        c = c.unsqueeze(dim=0).repeat(fg_sims.size(0),1,1)

        ## reshape

        sims = torch.zeros(bi, bt).to(device=caps.device)
        for i in range(fg_sims.size(0)):
            for j in range(fg_sims.size(1)):
                with torch.no_grad():
                    P = ot.emd(r[i,j],c[i,j],1-fg_sims[i,j])
                sims[i,j] = (P*fg_sims[i,j]).sum(dim=[-2,-1])
        return sims