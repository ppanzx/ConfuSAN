import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modules.utils import get_mask,get_fgsims,get_fgmask

EPS = 1e-8 # epsilon 
MASK = -1 # padding value
INF = -math.inf

## non-negative matrix factorization
class NMF(nn.Module):
    def __init__(self, tau=0.1, split=4, iters=5):
        super(NMF,self).__init__()
        self.split = split
        self.iters = iters
        self.eps = 1e-6
        self.tau = tau

    def nmf(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]
        imgs = F.normalize(imgs, p=2, dim=-1)
        caps = F.normalize(caps, p=2, dim=-1)

        img_mask = get_mask(img_lens) # Bi x K x 1
        cap_mask = get_mask(cap_lens) # Bt x L x 1

        sims = torch.einsum("ikd,tld->itkl", imgs, caps) # Bi x Bt x K x L
        mask = torch.einsum("ikd,tld->itkl", img_mask, cap_mask) # Bi x Bt x K x L
        sims = sims.masked_fill(mask == 0, 0)

        # initialize the attention matrix
        attn = sims.masked_fill(mask == 0, 0)

        eta = 2e-1/self.iters
        for i in range(self.iters):
            dnm = torch.einsum("imd,ind,itnl->itml", imgs, imgs, attn)+self.eps # denominator
            # attn = attn*sims/dnm
            attn = attn + eta*(sims-dnm)

            # attn = attn.masked_fill(mask == 0, 0)
            attn = attn.masked_fill(attn<0, 0)
            attn = attn/(attn.sum(dim=-2, keepdim=True)+self.eps) #

        # sims = torch.einsum("itkl,itkl->it",sims,attn)
        attn_caps = torch.einsum("itkl,ikd->itld", attn, imgs)
        attn_caps = F.normalize(attn_caps, p=2, dim=-1)
        sims = torch.einsum("itld,tld->itl", attn_caps, caps)
        sims = torch.einsum("itl,tli->it", sims, cap_mask)
        sims = sims/cap_lens.unsqueeze(dim=0)
        return sims
    
    def cam(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]
        imgs = F.normalize(imgs, p=2, dim=-1)
        caps = F.normalize(caps, p=2, dim=-1)

        img_mask = get_mask(img_lens) # Bi x K x 1
        cap_mask = get_mask(cap_lens) # Bt x L x 1

        sims = torch.einsum("ikd,tld->itkl", imgs, caps) # Bi x Bt x K x L
        mask = torch.einsum("ikd,tld->itkl", img_mask, cap_mask) # Bi x Bt x K x L

        attn = sims.masked_fill(sims < 0, INF)
        attn = attn.masked_fill(mask == 0, INF)
        attn = torch.softmax(attn/self.tau, dim=-2)
        attn = attn.masked_fill(mask == 0, 0)
        attn = attn.masked_fill(sims < 0, 0)

        attn_caps = torch.einsum("itkl,ikd->itld", attn, imgs)
        attn_caps = F.normalize(attn_caps, p=2, dim=-1)
        sims = torch.einsum("itld,tld->itl", attn_caps, caps)
        sims = torch.einsum("itl,tli->it", sims, cap_mask)
        sims = sims/cap_lens.unsqueeze(dim=0)

        return sims

    def subspace(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]
        imgs = F.normalize(imgs, p=2, dim=-1)
        caps = F.normalize(caps, p=2, dim=-1)

        img_mask = get_mask(img_lens) # Bi x K x 1
        cap_mask = get_mask(cap_lens) # Bt x L x 1

        sims = torch.einsum("ikd,tld->itkl", imgs, caps) # Bi x Bt x K x L
        mask = torch.einsum("ikd,tld->itkl", img_mask, cap_mask) # Bi x Bt x K x L
        sims = sims.masked_fill(mask == 0, 0)

        # initialize the attention matrix
        Bi = imgs.shape[0]
        Bt = caps.shape[0]
        img_mask = img_mask.repeat(1,1,Bt).permute(0,2,1)
        omega_1 = img_mask*(1/img_mask.sum(dim=-1, keepdim=True))
        omega_1 = omega_1.masked_fill(img_mask==0, 0)

        cap_mask = cap_mask.repeat(1,1,Bi).permute(2,0,1)
        omega_2 = cap_mask*(1/cap_mask.sum(dim=-1, keepdim=True))
        omega_2 = omega_2.masked_fill(cap_mask == 0, 0)

        eta = 1e-1/self.iters
        for i in range(self.iters):
            res_1 = torch.einsum("itl,tld->itd",omega_2, caps)-torch.einsum("itk,ikd->itd", omega_1, imgs)
            omega_2 = omega_2 - eta * torch.einsum("itd,tld->itl",res_1, caps)
            omega_2 = omega_2.masked_fill(cap_mask==0, 0)
            omega_2 = omega_2.masked_fill(omega_2<0, 0)
            omega_2 = omega_2/(omega_2.sum(dim=-1, keepdim=True)+self.eps)

            res_2 = torch.einsum("itl,tld->itd",omega_2, caps)-torch.einsum("itk,ikd->itd", omega_1, imgs)
            omega_1 = omega_1 + eta * torch.einsum("itd,ikd->itk",res_2, imgs)
            omega_1 = omega_1.masked_fill(img_mask==0, 0)
            omega_1 = omega_1.masked_fill(omega_1<0, 0)
            omega_1 = omega_1/(omega_1.sum(dim=-1, keepdim=True)+self.eps)

        # sims = torch.einsum("itkl,itkl->it",sims,attn)
        g_img = torch.einsum("itk,ikd->itd", omega_1, imgs)
        g_img = F.normalize(g_img, p=2, dim=-1)
        g_cap = torch.einsum("itl,tld->itd", omega_2, caps)
        g_cap = F.normalize(g_cap, p=2, dim=-1)
        sims = torch.einsum("itd,itd->it", g_img, g_cap)
        return sims
    
    def hausdorff(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]
        imgs = F.normalize(imgs, p=2, dim=-1)
        caps = F.normalize(caps, p=2, dim=-1)

        img_mask = get_mask(img_lens) # Bi x K x 1
        cap_mask = get_mask(cap_lens) # Bt x L x 1

        sims = torch.einsum("ikd,tld->itkl", imgs, caps) # Bi x Bt x K x L
        mask = torch.einsum("ikd,tld->itkl", img_mask, cap_mask) # Bi x Bt x K x L
        sims = sims.masked_fill(mask == 0, 0)

        # initialize the attention matrix
        Bi = imgs.shape[0]
        Bt = caps.shape[0]
        img_mask = img_mask.repeat(1,1,Bt).permute(0,2,1)
        omega_1 = img_mask*(1/img_mask.sum(dim=-1, keepdim=True))
        omega_1 = omega_1.masked_fill(img_mask==0, 0)

        cap_mask = cap_mask.repeat(1,1,Bi).permute(2,0,1)
        omega_2 = cap_mask*(1/cap_mask.sum(dim=-1, keepdim=True))
        omega_2 = omega_2.masked_fill(cap_mask == 0, 0)

        eta = 2e-1/self.iters
        for i in range(self.iters):
            res_1 = torch.einsum("itl,tld->itd",omega_2, caps)-torch.einsum("itk,ikd->itd", omega_1, imgs)
            omega_1 = omega_1 + eta * torch.einsum("itd,ikd->itk",res_1, imgs)
            omega_1 = omega_1.masked_fill(img_mask==0, 0)
            omega_1 = omega_1.masked_fill(omega_1<0, 0)
            omega_1 = omega_1/(omega_1.sum(dim=-1, keepdim=True)+self.eps)

            res_2 = torch.einsum("itl,tld->itd",omega_2, caps)-torch.einsum("itk,ikd->itd", omega_1, imgs)
            omega_2 = omega_2 + eta * torch.einsum("itd,tld->itl",res_2, caps)
            omega_2 = omega_2.masked_fill(cap_mask==0, 0)
            omega_2 = omega_2.masked_fill(omega_2<0, 0)
            omega_2 = omega_2/(omega_2.sum(dim=-1, keepdim=True)+self.eps)

        # sims = torch.einsum("itkl,itkl->it",sims,attn)
        g_img = torch.einsum("itk,ikd->itd", omega_1, imgs)
        g_img = F.normalize(g_img, p=2, dim=-1)
        g_cap = torch.einsum("itl,tld->itd", omega_2, caps)
        g_cap = F.normalize(g_cap, p=2, dim=-1)
        sims_1 = torch.einsum("itd,itd->it", g_img, g_cap)

        # initialize the attention matrix
        omega_1 = img_mask*(1/img_mask.sum(dim=-1, keepdim=True))
        omega_1 = omega_1.masked_fill(img_mask==0, 0)
        omega_2 = cap_mask*(1/cap_mask.sum(dim=-1, keepdim=True))
        omega_2 = omega_2.masked_fill(cap_mask == 0, 0)

        for i in range(self.iters):
            res_2 = torch.einsum("itl,tld->itd",omega_2, caps)-torch.einsum("itk,ikd->itd", omega_1, imgs)
            omega_2 = omega_2 - eta * torch.einsum("itd,tld->itl",res_2, caps)
            omega_2 = omega_2.masked_fill(cap_mask==0, 0)
            omega_2 = omega_2.masked_fill(omega_2<0, 0)
            omega_2 = omega_2/(omega_2.sum(dim=-1, keepdim=True)+self.eps)

            res_1 = torch.einsum("itl,tld->itd",omega_2, caps)-torch.einsum("itk,ikd->itd", omega_1, imgs)
            omega_1 = omega_1 - eta * torch.einsum("itd,ikd->itk",res_1, imgs)
            omega_1 = omega_1.masked_fill(img_mask==0, 0)
            omega_1 = omega_1.masked_fill(omega_1<0, 0)
            omega_1 = omega_1/(omega_1.sum(dim=-1, keepdim=True)+self.eps)

        # sims = torch.einsum("itkl,itkl->it",sims,attn)
        g_img = torch.einsum("itk,ikd->itd", omega_1, imgs)
        g_img = F.normalize(g_img, p=2, dim=-1)
        g_cap = torch.einsum("itl,tld->itd", omega_2, caps)
        g_cap = F.normalize(g_cap, p=2, dim=-1)
        sims_2 = torch.einsum("itd,itd->it", g_img, g_cap)
        sims = torch.max(sims_1, sims_2)
        return sims
    
    def forward(self, imgs, caps, img_lens, cap_lens):
        if not self.training:
            return self.subspace(imgs, caps, img_lens, cap_lens)
        n_image,_,_ = imgs.shape
        n_caption,_,_ = caps.shape
        sims = torch.zeros(n_image,n_caption).to(device=caps.device)
        step = n_caption//self.split
        for i in range(self.split):
            beg = step*i
            ed = n_caption if i+1==self.split else step*(i+1) 
            if beg>=ed:break
            sims[:,beg:ed] = self.subspace(imgs, caps[beg:ed], img_lens, cap_lens[beg:ed])
        return sims

# class NMF(nn.Module):
#     def __init__(self, iters=5):
#         super(NMF,self).__init__()
#         self.iters = iters
#         self.eps = 1e-6

#     def forward(self, imgs, caps, img_lens, cap_lens):
#         """
#         Images: (n_image, n_regions, d) matrix of images
#         Captions: (n_caption, max_n_word, d) matrix of captions
#         CapLens: (n_caption) array of caption lengths
#         """
#         sims = []
#         n_img = imgs.size(0)
#         n_cap = caps.size(0)
#         for i in range(n_cap):
#             # Get the i-th text description
#             n_word = int(cap_lens[i].item())
#             cap_i = caps[i, :n_word, :].unsqueeze(0).contiguous()
#             # --> (n_image, n_word, d)
#             cap_i_expand = cap_i.repeat(n_img, 1, 1)

#             sim = torch.einsum("ikd,ild->ikl", imgs, cap_i_expand, )

#             # initialize the attention matrix
#             attn = sim.clone()

#             for i in range(self.iters):
#                 dnm = torch.einsum("imd,ind->imn", imgs, imgs) # denominator
#                 dnm = torch.einsum("imn,ind->imd", dnm, attn)+self.eps
                
#                 attn = attn*sim/dnm
#                 attn[attn<0]=0
#                 attn = attn/(attn.sum(dim=-2, keepdim=True)+self.eps) #

#             attn_cap = torch.einsum("ikl,ikd->ild", attn, imgs, )
#             attn_cap = F.normalize(attn_cap, p=2, dim=-1)
#             col_sim = torch.einsum("ild,ild->il", attn_cap, cap_i_expand, )
#             col_sim = col_sim.mean(dim=1, keepdim=True)
#             sims.append(col_sim)

#         # (n_image, n_caption)
#         sims = torch.cat(sims, dim=1)
#         return sims