""" Hierarchic Optimal Transport module"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.aggr.pooling import AvePool
from lib.modules.utils import get_mask,get_fgsims,get_fgmask
from lib.modules.cluster import CTM,TCBlock
from lib.aggr.pvse import MultiHeadSelfAttention
from lib.modules.sparsemax import Sparsemax,sparsemax
INF = -1e3

# Hierarchical fine-grained sinkhorn distance
class HCAM(nn.Module):
    def __init__(self, iters=3, lamb=5e-2, split=4, _init=0):
        super(HCAM, self).__init__()
        self.eps = 1e-6
        self.iters = iters
        self.lamb = lamb
        self.split = split
        
        self._init = _init
        self.sa = Sparsemax()

    def _ave_init(self, mask):
        r = mask.sum(dim=[-1])!=0  # Bi x Bt x K
        c = mask.sum(dim=[-2])!=0  # Bi x Bt x L
        r = r*(1/r.sum(dim=-1, keepdim=True))
        c = c*(1/c.sum(dim=-1, keepdim=True))
        return r, c
    
    def forward(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        if self.training:
            bi, bt = imgs.size(0), caps.size(0)
            sims = torch.zeros(bi, bt).to(device=caps.device)
            step = bi//self.split
            for i in range(self.split):
                beg = step*i
                ed = bi if i+1==self.split else step*(i+1) 
                sims[beg:ed] = self.forward_s2d(img_cls[beg:ed], imgs[beg:ed], cap_cls, caps, img_lens[beg:ed], cap_lens,)
            return sims
        else: return self.forward_s2d(img_cls, imgs, cap_cls, caps, img_lens, cap_lens,)

    ## cross-attention method
    def forward_cam(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps

        imgs = imgs / imgs.norm(dim=-1, keepdim=True)
        caps = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs, caps)
        mask = get_fgmask(img_lens,cap_lens)

        v2t_attn = fg_sims.masked_fill(mask == 0, INF)/self.lamb
        v2t_attn = torch.softmax(v2t_attn,dim=-1)
        v2t = torch.einsum("itkl,itkl,itkl->itk",v2t_attn,fg_sims,mask)

        t2v_attn = fg_sims.masked_fill(mask == 0, INF)/self.lamb
        t2v_attn = torch.softmax(t2v_attn,dim=-2)
        t2v = torch.einsum("itkl,itkl,itkl->itl",t2v_attn,fg_sims,mask)

        r,c = self._ave_init(mask)
        v2t = torch.einsum("itk,itk->it",v2t,r)
        t2v = torch.einsum("itl,itl->it",t2v,c)
        sims = (t2v+v2t)/2
        return sims

    ## sparse-attention method
    def forward_sam(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps

        imgs = imgs / imgs.norm(dim=-1, keepdim=True)
        caps = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs, caps)
        mask = get_fgmask(img_lens,cap_lens)

        attn = fg_sims.masked_fill(mask == 0, -1)
        v2t_attn = sparsemax(attn)
        v2t = torch.einsum("itkl,itkl,itkl->itk",v2t_attn,fg_sims,mask)


        t2v_attn = sparsemax(attn.permute(0,1,3,2)).permute(0,1,3,2)
        t2v = torch.einsum("itkl,itkl,itkl->itl",t2v_attn,fg_sims,mask)

        r,c = self._ave_init(mask)
        v2t = torch.einsum("itk,itk->it",v2t,r)
        t2v = torch.einsum("itl,itl->it",t2v,c)
        sims = (t2v+v2t)/2
        return sims

    ## partial sparse-attention method
    def forward_hcam(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps

        imgs = torch.concat([img_cls.unsqueeze(dim=1), imgs], dim=1)
        caps = torch.concat([cap_cls.unsqueeze(dim=1), caps], dim=1)
        imgs = imgs / imgs.norm(dim=-1, keepdim=True)
        caps = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs, caps)
        img_lens = img_lens + 1
        cap_lens = cap_lens + 1

        mask = get_fgmask(img_lens, cap_lens)
        r,c = self._ave_init(mask)

        v2t_attn = fg_sims.masked_fill(mask == 0, INF)/self.lamb
        v2t_attn = torch.softmax(v2t_attn,dim=-1)
        v2t = torch.einsum("itkl,itkl,itkl->itk",v2t_attn,fg_sims,mask)

        t2v_attn = fg_sims.masked_fill(mask == 0, INF)/self.lamb
        t2v_attn = torch.softmax(t2v_attn,dim=-2)
        t2v = torch.einsum("itkl,itkl,itkl->itl",t2v_attn,fg_sims,mask)

        r,c = self._ave_init(mask)
        v2t = torch.einsum("itk,itk->it",v2t[:,:,1:],r[:,:,1:])
        t2v = torch.einsum("itl,itl->it",t2v[:,:,1:],c[:,:,1:])
        sims = (t2v+v2t)/2
        return sims

    ## sparsemax-2d method
    def forward_s2d(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps

        imgs = imgs / imgs.norm(dim=-1, keepdim=True)
        caps = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs, caps)
        mask = get_fgmask(img_lens,cap_lens)

        fg_sims = fg_sims.masked_fill(mask == 0, 0).reshape(bi, bt, -1)
        attn = sparsemax(fg_sims)
        sims = (fg_sims*attn).sum(dim=-1)
        return sims