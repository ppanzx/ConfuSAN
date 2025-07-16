"""Partial Optimal Transport module"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.aggr.pooling import AvePool
from lib.modules.utils import get_mask,get_fgsims,get_fgmask
INF = math.inf

# fine-grained sinkhorn distance
class Wasserstain(nn.Module):
    def __init__(self, iters=3, lamb=5e-2, split=4, _init=0):
        super(Wasserstain, self).__init__()
        self.eps = 1e-6
        self.iters = iters
        self.lamb = lamb
        self.split = split
        
        self._init = _init

    def _ave_init(self, mask):
        r = mask.sum(dim=[-1])!=0  # Bi x Bt x K
        c = mask.sum(dim=[-2])!=0  # Bi x Bt x L
        r = r*(1/r.sum(dim=-1, keepdim=True))
        c = c*(1/c.sum(dim=-1, keepdim=True))
        return r, c

    def _yokoi_init(self, features, mask):
        """
            <Word Rotator's Distance>
        """
        # max_len = int(lengths.max())
        # features = features[:,:max_len,:]
        weight = torch.norm(features,p=2,dim=-1,keepdim=True)
        weight = weight.masked_fill(mask == 0, 0)
        weight = weight/weight.sum(dim=1,keepdim=True)
        return weight

    def _zou_init(self, imgs, caps, img_lens, cap_lens,):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]
        
        Bi, _, _ = imgs.shape
        Bt, _, _ = caps.shape
        tau = 1

        pool = AvePool()
        img_glo, _ = pool(imgs, img_lens)
        img_glo = F.normalize(img_glo, dim=-1)
        cap_glo, _ = pool(caps, cap_lens)
        cap_glo = F.normalize(cap_glo, dim=-1)
        r = (imgs @ cap_glo.t()).permute(0, 2, 1)
        c = (caps @ img_glo.t()).permute(2, 0, 1)

        img_mask = get_mask(img_lens).permute(0, 2, 1).repeat(1, Bt, 1)
        cap_mask = get_mask(cap_lens).permute(2, 0, 1).repeat(Bi, 1, 1)

        r = r.masked_fill(img_mask == 0, -INF)
        r = torch.softmax(r/tau,dim=-1)
        c = c.masked_fill(cap_mask == 0, -INF)
        c = torch.softmax(c/tau,dim=-1)

        return r, c

    def _liu_init(self, features, length):
        """
            <Word Rotator’s Distance>
        """
        _max = int(length.max())
        features = features[:,:_max,:]
        pool = AvePool()
        
        _glo, _ = pool(features, length)
        _glo = F.normalize(_glo, dim=-1)
        tau = 1
        
        weight = torch.einsum('bd,bcd->bc', _glo, features).contiguous()
        # weight = features @ _glo.t()
        # weight = F.normalize(weight, dim=-1)
        _mask = get_mask(length)
        weight = weight.unsqueeze(dim=-1).masked_fill(_mask == 0, -INF)
        weight = torch.softmax(weight/tau, dim=1)
        # weight = weight.masked_fill(_mask == 0, 0)
        return weight

    def Sinkhorn_Knopp(self, sims, r, c):
        """
        Computes the optimal transport matrix and Slinkhorn distance using the
        Sinkhorn-Knopp algorithm
        """
        P = torch.exp(-1 / self.lamb * sims)    
        # Avoiding poor math condition
        P = P / (P.sum(dim=[-2,-1], keepdim=True)+self.eps)

        # Normalize this matrix so that P.sum(-1) == r, P.sum(-2) == c
        for i in range(self.iters):
            # Shape (n, )
            u = P.sum(dim=[-1],) + self.eps # u(0)
            P = P * (r / u).unsqueeze(dim=-1) # u(0)*
            v = P.sum(dim=[-2]) + self.eps
            P = P * (c / v).unsqueeze(dim=-2)
            # if (u - P.sum(dim=[-1],)).max()<self.eps or \
            #     (v - P.sum(dim=[-2],)).max()<self.eps:
            #     break
        return P

    def forward(self, imgs, caps, img_lens, cap_lens, return_attn=False):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        fg_sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)
        if return_attn: attn = torch.zeros_like(fg_sims,device=fg_sims.device)

        sims = torch.zeros(bi, bt).to(device=caps.device)
        step = bi//self.split
        for i in range(self.split):
            beg = step*i
            ed = bi if i+1==self.split else step*(i+1) 
            if self._init==0:
                r,c = self._ave_init(mask[beg:ed])
            elif self._init==1:
                img_mask = get_mask(img_lens)
                cap_mask = get_mask(cap_lens)
                r = self._yokoi_init(imgs[beg:ed,:int(img_lens.max())], img_mask[beg:ed])
                r = r.permute(0,2,1).repeat(1,caps.shape[0],1)
                c = self._yokoi_init(caps[:,:int(cap_lens.max())], cap_mask)
                c = c.permute(2,0,1).repeat(imgs[beg:ed].shape[0],1,1)
            elif self._init==2:
                r = self._liu_init(imgs[beg:ed,:int(img_lens.max())], img_lens[beg:ed])
                r = r.permute(0,2,1).repeat(1,caps.shape[0],1)
                c = self._liu_init(caps[:,:int(cap_lens.max())], cap_lens)
                c = c.permute(2,0,1).repeat(imgs[beg:ed].shape[0],1,1)
            elif self._init==3:
                r,c = self._zou_init(imgs[beg:ed], caps, img_lens[beg:ed], cap_lens)
            else:
                raise ValueError
            tp = self.Sinkhorn_Knopp((1-fg_sims[beg:ed]).masked_fill(mask[beg:ed] == 0, INF), r, c)
            sims[beg:ed] = (fg_sims[beg:ed]*tp*mask[beg:ed]).sum(dim=[-2,-1])
            if return_attn: attn[beg:ed]=tp
        if return_attn: return sims, attn
        else: return sims

if __name__=="__main__":
    batch_size = 128
    imgs = torch.rand([batch_size,36,1024])
    caps = torch.rand([batch_size,51,1024])
    img_lens = torch.randint(20, 37, [batch_size])
    cap_lens = torch.randint(36, 52, [batch_size])
    model = Wasserstain(yokoi_init=True)
    sims = model(imgs, caps, img_lens, cap_lens)