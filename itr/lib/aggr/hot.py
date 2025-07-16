""" Hierarchic Optimal Transport module"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.aggr.pooling import AvePool
from lib.modules.utils import get_mask,get_fgsims,get_fgmask
from lib.modules.cluster import CTM,TCBlock
from lib.aggr.pvse import MultiHeadSelfAttention
from lib.modules.utils import SelfAttention,Transformer
INF = math.inf

# Hierarchical fine-grained sinkhorn distance
class HOT(nn.Module):
    def __init__(self, iters=3, lamb=2e-2, split=4, _init=None):
        super(HOT, self).__init__()
        self.eps = 1e-6
        self.iters = iters
        self.lamb = lamb
        self.split = split
        
        self.aggr = eval("self.forward_%s"%_init)

        ratio_1 = 0.25
        ratio_2 = 0.5

        self.t_ctm0 = CTM(sample_ratio=ratio_1, embed_dim=1024, dim_out=1024, k=5)
        self.t_block0 = TCBlock(dim=1024, num_heads=8)
        self.t_ctm1 = CTM(sample_ratio=ratio_2, embed_dim=1024, dim_out=1024, k=3)
        self.t_block1 = TCBlock(dim=1024, num_heads=8)

        self.v_ctm0 = CTM(sample_ratio=ratio_1, embed_dim=1024, dim_out=1024, k=5)
        self.v_block0 = TCBlock(dim=1024, num_heads=8)
        self.v_ctm1 = CTM(sample_ratio=ratio_2, embed_dim=1024, dim_out=1024, k=3)
        self.v_block1 = TCBlock(dim=1024, num_heads=8)

        self.img_mhsa_0 = MultiHeadSelfAttention(8, 1024, 1024//2)
        self.cap_mhsa_0 = MultiHeadSelfAttention(8, 1024, 1024//2)

        self.img_mhsa_1 = MultiHeadSelfAttention(4, 1024, 1024//2)
        self.cap_mhsa_1 = MultiHeadSelfAttention(4, 1024, 1024//2)

        self.nctx = 1
        v_ctx = torch.empty(self.nctx,1024)
        t_ctx = torch.empty(self.nctx,1024)
        nn.init.normal_(v_ctx, std=0.01)   # define the prompt to be trained
        nn.init.normal_(t_ctx, std=0.01)   # define the prompt to be trained
        self.v_ctx = nn.Parameter(v_ctx)  # to be optimized
        self.t_ctx = nn.Parameter(t_ctx)  # to be optimized

        self.z = nn.Parameter(torch.ones([]) + 2e-1)
        self.sa = SelfAttention(1024)

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

    def _zou_init(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
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
            <Word Rotator's Distance>
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

    # @torch.no_grad()
    def Bregman(self, sims, r, c):
        """
        Computes the optimal transport matrix and Slinkhorn distance using the
        Sinkhorn-Knopp algorithm
        """
        P = torch.exp(sims / self.lamb)    
        # Avoiding poor math condition
        P = P / (P.sum(dim=[-2,-1], keepdim=True)+self.eps)

        # Normalize this matrix so that P.sum(-1) == r, P.sum(-2) == c
        for i in range(self.iters):
            # Shape (n, )
            # P0 = P
            u = P.sum(dim=[-1],) + self.eps # u(0)
            P = P * (r / u).unsqueeze(dim=-1) # u(0)*
            v = P.sum(dim=[-2]) + self.eps
            P = P * (c / v).unsqueeze(dim=-2)
            # err = (P0 - P).abs().mean()
            # if err.item()<self.eps:
            #     break
        return P

    ## TODO: NaN-related issues arise for unknown reasons
    def Sinkhorn_Knopp(self, sims, r, c):
        P = torch.exp(-1 / self.lamb * sims) 
        P = P / (P.sum(dim=[-2,-1], keepdim=True)+self.eps)
        v = torch.ones_like(r)
        
        # Normalize this matrix so that P.sum(-1) == r, P.sum(-2) == c
        for i in range(self.iters):
            v0 = v
            u = torch.einsum("itkl,itk->itl",P,v)
            u = c/u.masked_fill(c==0,INF)
            v = torch.einsum("itkl,itl->itk",P,u)
            v = r/v.masked_fill(r==0,INF)
            # err = (v - v0).abs().mean()
            # if err.item() < self.eps:
            #     break
        # P = u.unsqueeze(dim=-2)*v.unsqueeze(dim=-1)*P
        P = torch.einsum("itk,itkl,itl->itkl",v,P,u)
        return P

    def forward(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        if self.training:
            bi, bt = imgs.size(0), caps.size(0)
            sims = torch.zeros(bi, bt).to(device=caps.device)
            step = bi//self.split
            for i in range(self.split):
                beg = step*i
                ed = bi if i+1==self.split else step*(i+1) 
                sims[beg:ed] = self.aggr(img_cls[beg:ed], imgs[beg:ed], cap_cls, caps, img_lens[beg:ed], cap_lens,)
            return sims
        else: return self.aggr(img_cls, imgs, cap_cls, caps, img_lens, cap_lens,)

    def forward_pot(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps
        img_mask = get_mask(img_lens)
        cap_mask = get_mask(cap_lens)

        ## action level
        imgs_a,_ = self.img_mhsa_0(imgs, img_mask.squeeze())
        caps_a,_ = self.cap_mhsa_0(caps, cap_mask.squeeze())

        ## event level
        imgs_e,_ = self.img_mhsa_1(imgs_a)
        caps_e,_ = self.cap_mhsa_1(caps_a)

        imgs_all = torch.concat([imgs_e, imgs_a, imgs], dim=1)
        caps_all = torch.concat([caps_e, caps_a, caps], dim=1)
        img_lens = img_lens + imgs_a.size(1) + imgs_e.size(1)
        cap_lens = cap_lens + caps_a.size(1) + caps_e.size(1)

        fg_sims = get_fgsims(imgs_all, caps_all)
        mask = get_fgmask(img_lens, cap_lens)
        fg_sims = fg_sims.masked_fill(mask == 0, -1)

        r,c = self._ave_init(mask)
        tp = self.Bregman((1-fg_sims).masked_fill(mask == 0, INF), r, c)
        sims = (fg_sims*tp*mask).sum(dim=[-2,-1])
        return sims
    
    ## optimal patial transport-full
    def forward_optf(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:] + self.eps
        caps = caps[:,:max_w,:] + self.eps

        imgs = torch.concat([img_cls.unsqueeze(dim=1), imgs], dim=1)
        caps = torch.concat([cap_cls.unsqueeze(dim=1), caps], dim=1)
        
        imgs = imgs / imgs.norm(dim=-1, keepdim=True)
        caps = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs, caps)
        img_lens = img_lens + 1
        cap_lens = cap_lens + 1
        mask = get_fgmask(img_lens,cap_lens)
        r,c = self._ave_init(mask)
        tp = self.Bregman((1-fg_sims).masked_fill(mask == 0, INF), r, c)
        sims = (fg_sims*tp*mask).sum(dim=[-2,-1])
        return sims
    
    ## optimal patial transport-prompt
    def forward_optp(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps

        imgs = torch.concat([self.v_ctx.unsqueeze(dim=0).repeat(bi,1,1), imgs], dim=1)
        caps = torch.concat([self.t_ctx.unsqueeze(dim=0).repeat(bt,1,1), caps], dim=1)
        imgs = imgs / imgs.norm(dim=-1, keepdim=True)
        caps = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs, caps)
        img_lens = img_lens + 1
        cap_lens = cap_lens + 1

        mask = get_fgmask(img_lens, cap_lens)
        r,c = self._ave_init(mask)

        ## entity level
        tp = self.Bregman((1-fg_sims).masked_fill(mask == 0, INF), r, c)
        sims_hiot = (fg_sims*tp*mask)[:,:,1:,1:].sum(dim=[-2,-1])
        return sims_hiot

    ## optimal patial transport-dummy
    def forward_optd(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps

        imgs = imgs / imgs.norm(dim=-1, keepdim=True)
        caps = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs, caps)
        dummy_col = self.z+torch.zeros(1,1,max_r,1).repeat(bi,bt,1,1).to(imgs.device)
        fg_sims = torch.concat([dummy_col,fg_sims],dim=-1)
        dummy_row = self.z+torch.zeros(1,1,1,max_w+1).repeat(bi,bt,1,1).to(imgs.device)
        fg_sims = torch.concat([dummy_row,fg_sims],dim=-2)
        img_lens = img_lens + 1
        cap_lens = cap_lens + 1

        mask = get_fgmask(img_lens, cap_lens)
        r,c = self._ave_init(mask)

        ## entity level
        tp = self.Bregman((1-fg_sims).masked_fill(mask == 0, INF), r, c)
        sims_hiot = (fg_sims*tp*mask)[:,:,1:,1:].sum(dim=[-2,-1])
        return sims_hiot

    def forward_hiot(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
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

        ## entity level
        tp = self.Bregman((1-fg_sims).masked_fill(mask == 0, INF), r, c)
        sims_hiot = (fg_sims*tp*mask)[:,:,1:,1:].sum(dim=[-2,-1])
        return sims_hiot

    ## skip connection OT
    def forward_ot(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps
        imgs = imgs / imgs.norm(dim=-1, keepdim=True)
        caps = caps / caps.norm(dim=-1, keepdim=True)
        
        fg_sims = get_fgsims(imgs, caps)
        mask = get_fgmask(img_lens,cap_lens)
        r,c = self._ave_init(mask)
        tp = self.Bregman(fg_sims.masked_fill(mask == 0, -INF), r, c)
        sims = (fg_sims*tp*mask).sum(dim=[-2,-1])
        return sims

    ## mutual information
    def forward_mi(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps
        imgs = imgs / imgs.norm(dim=-1, keepdim=True)
        caps = caps / caps.norm(dim=-1, keepdim=True)
        
        fg_sims = get_fgsims(imgs, caps)
        mask = get_fgmask(img_lens,cap_lens)
        r,c = self._ave_init(mask)
        Pvt = self.Bregman((1-fg_sims).masked_fill(mask == 0, INF), r, c)
        # PvPt = torch.einsum("itk,itl->itkl",r,c)

        # sims = (Pvt*torch.log(Pvt.masked_fill(mask == 0, 1))).sum(dim=[-2,-1])
        # sims = sims+torch.log(img_lens).unsqueeze(dim=-1)+torch.log(cap_lens).unsqueeze(dim=0)

        sims = (fg_sims*Pvt*mask).sum(dim=[-2,-1])
        ent = (Pvt*torch.log(Pvt.masked_fill(mask == 0, 1))).sum(dim=[-2,-1])
        sims = sims + self.lamb*ent
        return sims

    def forward_otot(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        sims = self.forward_ot(imgs, caps, img_lens, cap_lens,)
        # ones = torch.ones_like(sims).to(sims.device)
        r = torch.ones(sims.size(0)).to(sims.device) # Bi
        c = torch.ones(sims.size(1)).to(sims.device) # Bi
        tp = self.Bregman((1-sims), r, c)
        return sims*tp

    def forward_mhot(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        d = imgs.shape[-1]
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps

        num_heads = 4
        imgs = imgs.reshape(bi, max_r, num_heads, d//num_heads)
        caps = caps.reshape(bt, max_w, num_heads, d//num_heads)
        imgs = imgs / imgs.norm(dim=-1, keepdim=True)
        caps = caps / caps.norm(dim=-1, keepdim=True)

        fg_sims = torch.einsum("vknd,tlnd->vtnkl",[imgs, caps])
        mask = get_fgmask(img_lens, cap_lens)
        mask = mask.unsqueeze(dim=2).repeat(1,1,num_heads,1,1)

        r,c = self._ave_init(mask)
        cost = (1-fg_sims).masked_fill(mask == 0, INF)
        tp = self.Bregman(cost, r, c) # Bi x Bt x heads x K x L
        sims = (fg_sims*tp*mask).sum(dim=[-2,-1])
        sims = sims.mean(dim=-1)
        return sims

    def forward_mhsa(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:] + self.eps
        caps = caps[:,:max_w,:] + self.eps
        img_mask = get_mask(img_lens)
        cap_mask = get_mask(cap_lens)

        imgs_ = imgs / imgs.norm(dim=-1, keepdim=True)
        caps_ = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs_, caps_)
        mask = get_fgmask(img_lens,cap_lens)

        sims = torch.zeros(3, bi, bt).to(device=caps.device)
        # entity level
        r,c = self._ave_init(mask)
        tp = self.Bregman((1-fg_sims).masked_fill(mask == 0, INF), r, c)
        sims[0] = (fg_sims*tp*mask).sum(dim=[-2,-1])

        # action level
        imgs,_ = self.img_mhsa_0(imgs, img_mask.squeeze())
        caps,_ = self.cap_mhsa_0(caps, cap_mask.squeeze())
        imgs_ = imgs / imgs.norm(dim=-1, keepdim=True)
        caps_ = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs_, caps_)
        mask = get_fgmask(torch.zeros(imgs_.size(0))+imgs_.size(1),
                    torch.zeros(caps_.size(0))+caps_.size(1)).to(imgs_.device)
        # cap_lens = torch.zeros(caps.size(0))+caps.size(1)
        # mask = get_fgmask(img_lens,cap_lens.to(img_lens.device)).to(imgs.device)
        r,c = self._ave_init(mask)
        tp = self.Bregman((1-fg_sims).masked_fill(mask == 0, INF), r, c)
        sims[1] = (fg_sims*tp*mask).sum(dim=[-2,-1])

        # event level
        imgs,_ = self.img_mhsa_1(imgs)
        caps,_ = self.cap_mhsa_1(caps)

        imgs_ = imgs / imgs.norm(dim=-1, keepdim=True)
        caps_ = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs_, caps_)
        mask = get_fgmask(torch.zeros(imgs_.size(0))+imgs_.size(1),
                    torch.zeros(caps_.size(0))+caps_.size(1)).to(imgs_.device)
        r,c = self._ave_init(mask)
        tp = self.Bregman((1-fg_sims).masked_fill(mask == 0, INF), r, c)
        sims[2] = (fg_sims*tp*mask).sum(dim=[-2,-1])
        # sims[2] = torch.einsum("vd,td->vt",[imgs,caps])
        return sims

    def forward_cluster(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:] + self.eps 
        caps = caps[:,:max_w,:] + self.eps
        img_mask = get_mask(img_lens)
        cap_mask = get_mask(cap_lens)

        imgs_ = imgs / imgs.norm(dim=-1, keepdim=True)
        caps_ = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs_, caps_)
        mask = get_fgmask(img_lens,cap_lens)

        sims = torch.zeros(3, bi, bt).to(device=caps.device)

        t_idx_token = torch.arange(caps.size(1))[None, :].repeat(caps.size(0), 1)
        t_agg_weight = caps.new_ones(caps.size(0), caps.size(1), 1)
        t_token_dict = {'x': caps,
                        'token_num': caps.size(1),
                        'idx_token': t_idx_token,
                        'agg_weight': t_agg_weight,
                        'mask': cap_mask.squeeze().detach()}
        
        v_idx_token = torch.arange(imgs.size(1))[None, :].repeat(imgs.size(0), 1)
        v_agg_weight = imgs.new_ones(imgs.size(0), imgs.size(1), 1)
        v_token_dict = {'x': imgs,
                        'token_num': imgs.size(1),
                        'idx_token': v_idx_token,
                        'agg_weight': v_agg_weight,
                        'mask': img_mask.squeeze().detach()}

        # entity level
        r,c = self._ave_init(mask)
        tp = self.Bregman((1-fg_sims).masked_fill(mask == 0, INF), r, c)
        sims[0] = (fg_sims*tp*mask).sum(dim=[-2,-1])

        # action level
        t_token_dict = self.t_block0(self.t_ctm0(t_token_dict))
        v_token_dict = self.v_block0(self.v_ctm0(v_token_dict))
        caps = t_token_dict["x"]
        imgs = v_token_dict["x"]

        imgs_ = imgs / imgs.norm(dim=-1, keepdim=True)
        caps_ = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs_, caps_)
        mask = get_fgmask(torch.zeros(imgs_.size(0))+imgs_.size(1),
                    torch.zeros(caps_.size(0))+caps_.size(1)).to(imgs_.device)
        # cap_lens = torch.zeros(caps.size(0))+caps.size(1)
        # mask = get_fgmask(img_lens,cap_lens.to(img_lens.device)).to(imgs.device)
        r,c = self._ave_init(mask)
        tp = self.Bregman((1-fg_sims).masked_fill(mask == 0, INF), r, c)
        sims[1] = (fg_sims*tp*mask).sum(dim=[-2,-1])

        # event level
        t_token_dict = self.t_block1(self.t_ctm1(t_token_dict))
        v_token_dict = self.v_block1(self.v_ctm1(v_token_dict))
        caps = t_token_dict["x"]
        imgs = v_token_dict["x"]

        imgs_ = imgs / imgs.norm(dim=-1, keepdim=True)
        caps_ = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs_, caps_)
        mask = get_fgmask(torch.zeros(imgs_.size(0))+imgs_.size(1),
                    torch.zeros(caps_.size(0))+caps_.size(1)).to(imgs_.device)
        # cap_lens = torch.zeros(caps.size(0))+caps.size(1)
        # mask = get_fgmask(img_lens,cap_lens.to(img_lens.device)).to(imgs.device)
        r,c = self._ave_init(mask)
        tp = self.Bregman((1-fg_sims).masked_fill(mask == 0, INF), r, c)
        sims[2] = (fg_sims*tp*mask).sum(dim=[-2,-1])
        return sims


