import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modules.utils import get_mask,get_fgsims,get_fgmask

EPS = 1e-8 # epsilon 
MASK = -1 # padding value
INF = -math.inf

## non-negative matrix factorization
class Hausdorff(nn.Module):
    def __init__(self, ):
        super(Hausdorff,self).__init__()

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)
        sims = sims.masked_fill(mask == 0, MASK)
        t2i = sims.max(dim=-2)[0]
        t2i[t2i==MASK] = -MASK
        t2i = t2i.min(dim=-1)[0]

        i2t = sims.max(dim=-1)[0]
        i2t[i2t==MASK] = -MASK
        i2t = i2t.min(dim=-1)[0]

        sims = torch.min(t2i, i2t)
        return sims
