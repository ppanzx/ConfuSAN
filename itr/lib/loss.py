import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
INF = 1e3

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt=None, margin=0.2, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        if opt is not None:
            self.opt = opt
            self.margin = opt.margin
            self.max_violation = opt.max_violation
        else:
            self.margin = margin
            self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, sims, mask=None):
        # compute image-sentence score matrix
        # sims = get_sim(im, s)
        diagonal = sims.diag().view(sims.size(0), 1)
        d1 = diagonal.expand_as(sims)
        d2 = diagonal.t().expand_as(sims)

        # compare every diagonal score to sims in its column
        # caption retrieval
        cost_s = (self.margin + sims - d1).clamp(min=0)
        # compare every diagonal score to sims in its row
        # image retrieval
        cost_im = (self.margin + sims - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(sims.size(0)) > .5
        I = Variable(mask).to(device=sims.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

class InfoNCELoss(nn.Module):
    """
    Compute InfoNCELoss loss
    """
    def __init__(self, temperature=0.01, margin=0):
        super(InfoNCELoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, sims, mask=None):
        ## cost of image retrieval
        img_ret = sims-sims.diag().expand_as(sims).t()+self.margin
        img_ret[torch.eye(sims.size(0))>.5] = 0
        cost_im = torch.log(torch.sum(torch.exp(img_ret/self.temperature),dim=1))

        ## cost of text retrieval
        txt_ret = sims-sims.diag().expand_as(sims)+self.margin
        txt_ret[torch.eye(sims.size(0))>.5] = 0
        cost_s = torch.log(torch.sum(torch.exp(txt_ret/self.temperature),dim=0))

        return cost_s.mean() + cost_im.mean()

    def max_violation_on(self):
        return 

    def max_violation_off(self):
        return

class HubnormTripletLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt=None, margin=0.2, max_violation=False):
        super(HubnormTripletLoss, self).__init__()
        if opt is not None:
            self.opt = opt
            self.margin = opt.margin
            self.max_violation = opt.max_violation
        else:
            self.margin = margin
            self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def ot(self, sims):
        ## hyperparameters
        iters = 5
        lamb = 0.012
        eps = INF
        r = torch.ones(sims.size(0)).to(sims.device)
        c = torch.ones(sims.size(1)).to(sims.device)

        P = torch.exp(-1 / lamb * (1-sims))    
        # Avoiding poor math condition
        P = P / (P.sum(dim=[-2,-1], keepdim=True))

        # Normalize this matrix so that P.sum(-1) == r, P.sum(-2) == c
        for i in range(iters):
            # Shape (n, )
            # P0 = P
            u = P.sum(dim=[-1],) # u(0)
            P = P * (r / u).unsqueeze(dim=-1) # u(0)*
            v = P.sum(dim=[-2]) 
            P = P * (c / v).unsqueeze(dim=-2)
            # err = (P0 - P).abs().mean()
            # if err.item()<self.eps:
            #     break
        # P = P.to(dtype)
        return P

    def forward(self, sims):
        # normlization
        sims = self.ot(sims)

        # image retrieval
        cost_im = sims-sims.diag().expand_as(sims)+self.margin
        cost_im[torch.eye(sims.size(0))>.5] = 0
        cost_im[cost_im<0] = 0

        # caption retrieval
        cost_s = sims-sims.diag().expand_as(sims).t()+self.margin
        cost_s[torch.eye(sims.size(0))>.5] = 0
        cost_s[cost_s<0] = 0

        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
    
class HubnormInfoNCELoss(nn.Module):
    """
    Compute InfoNCELoss loss
    """
    def __init__(self, temperature=0.01, margin=0):
        super(HubnormInfoNCELoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, sims):
        ## cost of image retrieval
        sims_t2i = sims * torch.softmax(sims/1, dim=0)
        img_ret = sims_t2i-sims_t2i.diag().expand_as(sims).t()+self.margin
        img_ret[torch.eye(sims.size(0))>.5] = 0
        cost_im = torch.log(torch.sum(torch.exp(img_ret/self.temperature),dim=1))

        ## cost of text retrieval
        sims_i2t = sims * torch.softmax(sims/0.01, dim=1)
        txt_ret = sims_i2t-sims_i2t.diag().expand_as(sims)+self.margin
        txt_ret[torch.eye(sims.size(0))>.5] = 0
        cost_s = torch.log(torch.sum(torch.exp(txt_ret/self.temperature),dim=0))

        return cost_s.mean() + cost_im.mean()

    def max_violation_on(self):
        return 

    def max_violation_off(self):
        return

class MixLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """
    def __init__(self, margin=0.05, temperature=0.01, max_violation=False):
        super(MixLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use InfoNCE objective.')

    def forward(self, sims, mask=None):
        if mask is None and sims.shape[0]==sims.shape[1]:
            mask = torch.eye(*sims.shape, device=sims.device)
        elif mask is None and sims.shape[0]!=sims.shape[1]:
            raise ValueError
        else: mask = mask.to(device=sims.device)
        if self.max_violation:
            # vsepp
            ## cost of image retrieval: image2text
            ep = sims.masked_fill(mask==0, INF).min(dim=0)[0] # easiest positive
            hn = sims.masked_fill(mask==1, -INF).max(dim=0)[0] # hardest positive
            cost_im = hn - ep + self.margin
            cost_im[cost_im<0] = 0

            ## cost of text retrieval
            ep = sims.masked_fill(mask==0, INF).min(dim=1)[0] # easiest positive
            hn = sims.masked_fill(mask==1, -INF).max(dim=1)[0] # hardest positive
            cost_t = hn - ep + self.margin
            cost_t[cost_t<0] = 0

            return cost_t.sum() + cost_im.sum()
        else:
            # infonce
            ## cost of image retrieval
            i2t = torch.softmax(sims/self.temperature,dim=0)
            cost_im = -((i2t*mask).sum(dim=0)).log()

            ## cost of text retrieval
            t2i = torch.softmax(sims/self.temperature,dim=1)
            cost_t = -((t2i*mask).sum(dim=1)).log()

            return cost_im.mean() + cost_t.mean()


class HierachicalLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """
    def __init__(self, lamb=0.05):
        super(HierachicalLoss, self).__init__()
        self.lamb = lamb

    def forward(self, sims, mask=None):
        self.margin = 0.2
        loss1 = self._forward(sims[0])
        self.margin = 0.05
        loss2 = self._forward(sims[1])
        return loss1+self.lamb*loss2
    
    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use InfoNCE objective.')

    def _forward(self, sims, mask=None):
        if mask is None and sims.shape[0]==sims.shape[1]:
            mask = torch.eye(*sims.shape, device=sims.device)
        elif mask is None and sims.shape[0]!=sims.shape[1]:
            raise ValueError
        else: mask = mask.to(device=sims.device)

        if self.max_violation:
            # vsepp
            ## cost of image retrieval: image2text
            ep = sims.masked_fill(mask==0, 1e5).min(dim=0)[0] # easiest positive
            hn = sims.masked_fill(mask==1, -1e5).max(dim=0)[0] # hardest positive
            cost_im = hn - ep + self.margin
            cost_im[cost_im<0] = 0

            ## cost of text retrieval
            ep = sims.masked_fill(mask==0, 1e5).min(dim=1)[0] # easiest positive
            hn = sims.masked_fill(mask==1, -1e5).max(dim=1)[0] # hardest positive
            cost_t = hn - ep + self.margin
            cost_t[cost_t<0] = 0

            return cost_t.sum() + cost_im.sum()
        
def get_criterion(criterion,opt,**args):
    if criterion=="ContrastiveLoss":
        return ContrastiveLoss(margin=opt.margin)
    elif criterion=="InfoNCELoss":
        return InfoNCELoss(temperature=opt.temperature,
                            margin=opt.margin)
    elif criterion=="MixLoss":
        return MixLoss(temperature=opt.temperature,
                            margin=opt.margin)
    elif criterion=="HierachicalLoss":
        return HierachicalLoss(lamb=opt.alpha,)
    else:
        raise ValueError("Unknown criterion type: {}".format(criterion))