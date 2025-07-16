import math
import torch
import torch.nn as nn
from lib.modules.sparsemax import sparsemax
from lib.modules.utils import get_mask, get_fgsims, get_fgmask
INF = 1e3

class Select(nn.Module):
    def __init__(self, mode=0):
        super(Select,self).__init__()
        if mode==0:
            self.select = self.max
        elif mode==1:
            self.select = self.sparsemax
        elif mode==2:
            self.select = self.softmax
        elif mode==3:
            self.select = self.dualsparsemax
        elif mode==4:
            self.select = self.logsumexp

    def marginal(self, vmask, tmask):
        bv = vmask.shape[0]
        bt = tmask.shape[0]

        vw = vmask/vmask.sum(dim=1,keepdims=True)
        vw = vw.repeat(1,1,bt).permute(0,2,1)
        tw = tmask/tmask.sum(dim=1,keepdims=True)
        tw = tw.repeat(1,1,bv).permute(2,0,1)
        return vw, tw

    def max(self, sims, mask):
        sims = sims.masked_fill(mask==0, -INF)

        r2w = sims.max(dim=-1)[0]
        r2w = r2w.masked_fill(r2w==-INF, 0)
        v2t = torch.einsum("itk,itk->it",r2w,self.vw)

        w2r = sims.max(dim=-2)[0]
        w2r = w2r.masked_fill(w2r==-INF, 0)
        t2v = torch.einsum("itl,itl->it",w2r,self.tw)

        sims = (v2t+t2v)/2
        return sims

    def sparsemax(self, sims, mask):
        sims = sims.masked_fill(mask==0, -1)

        r2w = sparsemax(sims, dim=-1)
        r2w = r2w.masked_fill(r2w==-1, 0)
        r2w = (r2w*sims).sum(dim=-1)
        v2t = torch.einsum("itk,itk->it", r2w, self.vw)

        w2r = sparsemax(sims.permute(0,1,3,2), dim=-1).permute(0,1,3,2)
        w2r = w2r.masked_fill(w2r==-1, 0)
        w2r = (w2r*sims).sum(dim=-2)
        t2v = torch.einsum("itl,itl->it", w2r, self.tw)

        sims = (v2t+t2v)/2
        return sims
    
    def softmax(self, sims, mask):
        tau = 0.1
        sims = sims.masked_fill(mask==0, -INF)

        r2w = torch.softmax(sims/tau, dim=-1)
        r2w = r2w.masked_fill(mask==0, 0)
        r2w = (r2w*sims).sum(dim=-1)
        v2t = torch.einsum("itk,itk->it",r2w,self.vw)

        w2r = torch.softmax(sims/tau, dim=-2)
        w2r = w2r.masked_fill(mask==0, 0)
        w2r = (w2r*sims).sum(dim=-2)
        t2v = torch.einsum("itl,itl->it",w2r,self.tw)

        sims = (v2t+t2v)/2
        return sims
     
    def dualsparsemax(self, sims, mask):
        sims = sims.masked_fill(mask==0, -1)

        r2w_attn = sparsemax(sims, dim=-1)
        r2w_attn = r2w_attn.masked_fill(r2w_attn==-1, 0)
        r2w = (r2w_attn*sims).sum(dim=-1)
        v2t_attn = sparsemax(r2w, dim=-1)
        v2t = (v2t_attn*r2w).sum(dim=-1)

        w2r_attn = sparsemax(sims.permute(0,1,3,2), dim=-1).permute(0,1,3,2)
        w2r_attn = w2r_attn.masked_fill(w2r_attn==-1, 0)
        w2r = (w2r_attn*sims).sum(dim=-2)
        t2v_attn = sparsemax(w2r, dim=-1)
        t2v = (t2v_attn*w2r).sum(dim=-1)

        sims = (v2t+t2v)/2
        return sims

    ## bug: value of logsumexp is too large
    def maxtoken(self, sims, mask):
        sims = sims.masked_fill(mask==0, -1)
        return sims

    def forward(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens):
        bi, bt = imgs.shape[0], caps.shape[0]
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]

        vmask = get_mask(img_lens)
        tmask = get_mask(cap_lens)
        self.vw, self.tw = self.marginal(vmask, tmask)

        fgmask = torch.einsum("ikd,tld->itkl", vmask, tmask)
        fgsims = torch.einsum("ikd,tld->itkl", imgs, caps)

        sims = self.select(fgsims, fgmask)
        return sims
