""" Optimal Transport module"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.aggr.pooling import AvePool
from lib.modules.utils import get_mask
INF = math.inf

# fine-grained sinkhorn distance
class Tokenflow(nn.Module):
    def __init__(self, tau=5e-2, mode=0):
        super(Tokenflow, self).__init__()
        self.tau = tau
        self.mode = mode

    def forward(self, imgs, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]

        pool = AvePool()
        img_glo, _ = pool(imgs, img_lens)
        img_glo = F.normalize(img_glo, dim=-1)
        cap_glo, _ = pool(caps, cap_lens)
        cap_glo = F.normalize(cap_glo, dim=-1)
        ds = torch.einsum('vkd,td->vtk', imgs, cap_glo).contiguous()
        et = torch.einsum('tld,vd->vtl', caps, img_glo).contiguous()

        sim = torch.einsum('vkd,tld->vtkl', imgs, caps).contiguous()
        img_mask = get_mask(img_lens)
        cap_mask = get_mask(cap_lens)
        sim_mask = torch.einsum('vkd,tld->vtkl', img_mask, cap_mask).contiguous()
        sim = sim.masked_fill(sim_mask == 0, -INF)

        if self.mode==0:
            w = torch.einsum("vtkl,vtl->vtkl", sim, et).contiguous()
            w = F.softmax(w/self.tau,dim=-1)
            w = torch.einsum("vtkl,vtk->vtkl", sim, ds).contiguous()
            w = sim.masked_fill(sim_mask == 0, 0)
        elif self.mode==1:
            w = torch.einsum("vtkl,vtk->vtkl", sim, ds).contiguous()
            w = F.softmax(w/self.tau,dim=-2)
            w = torch.einsum("vtkl,vtl->vtkl", sim, et).contiguous()
            w = sim.masked_fill(sim_mask == 0, 0)
        else: raise ValueError
        sim = torch.einsum("vtkl,vtkl->vt", sim, w).contiguous()

        return sim

if __name__=="__main__":
    batch_size = 128
    imgs = torch.rand([batch_size,36,1024])
    caps = torch.rand([batch_size,51,1024])
    img_lens = torch.randint(20, 37, [batch_size])
    cap_lens = torch.randint(36, 52, [batch_size])
    model = Tokenflow()
    sims = model(imgs, caps, img_lens, cap_lens)
    print(sims.shape)