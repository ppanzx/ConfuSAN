""" Hierarchic Optimal Transport module"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.aggr.pooling import AvePool
from lib.modules.utils import get_mask,get_fgsims,get_fgmask
from lib.modules.cluster import CTM,TCBlock
from lib.aggr.pvse import MultiHeadSelfAttention
INF = math.inf

# Hierarchical fine-grained sinkhorn distance
class HiCHAN(nn.Module):
    def __init__(self, iters=3, lamb=5e-2, split=4, _init=None):
        super(HiCHAN, self).__init__()
        self.eps = 1e-6
        self.iters = iters
        self.lamb = lamb
        self.split = split
        
        self._init = _init
        self.aggr = eval("self.forward_%s"%_init)

        self.t_ctm0 = CTM(sample_ratio=0.25, embed_dim=1024, dim_out=1024, k=3)
        self.t_block0 = TCBlock(dim=1024, num_heads=8)
        self.t_ctm1 = CTM(sample_ratio=0.5, embed_dim=1024, dim_out=1024, k=3)
        self.t_block1 = TCBlock(dim=1024, num_heads=8)

        self.v_ctm0 = CTM(sample_ratio=0.25, embed_dim=1024, dim_out=1024, k=3)
        self.v_block0 = TCBlock(dim=1024, num_heads=8)
        self.v_ctm1 = CTM(sample_ratio=0.5, embed_dim=1024, dim_out=1024, k=3)
        self.v_block1 = TCBlock(dim=1024, num_heads=8)

        self.img_mhsa_0 = MultiHeadSelfAttention(16, 1024, 1024//2)
        self.cap_mhsa_0 = MultiHeadSelfAttention(16, 1024, 1024//2)

        self.img_mhsa_1 = MultiHeadSelfAttention(4, 1024, 1024//2)
        self.cap_mhsa_1 = MultiHeadSelfAttention(4, 1024, 1024//2)

        self.nctx = 4
        v_ctx = torch.empty(self.nctx,1024)
        t_ctx = torch.empty(self.nctx,1024)
        nn.init.normal_(v_ctx, std=0.02)   # define the prompt to be trained
        nn.init.normal_(t_ctx, std=0.02)   # define the prompt to be trained
        self.v_ctx = nn.Parameter(v_ctx)  # to be optimized
        self.t_ctx = nn.Parameter(t_ctx)  # to be optimized

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

    def _chan_init(self, fg_sims, mask):
        fg_sims = fg_sims.masked_fill(mask == 0, -1)
        v2t_index = fg_sims.argmax(dim=-1, keepdim=True)
        t2v_index = fg_sims.argmax(dim=-2, keepdim=True)
        v2t_mask = v2t_index.repeat(1,1,1,fg_sims.shape[-1])==torch.arange(fg_sims.shape[-1]).expand_as(fg_sims)
        t2v_mask = t2v_index.repeat(1,1,fg_sims.shape[-2],1)==torch.arange(fg_sims.shape[-2]).expand_as(fg_sims.permute(0,1,3,2)).permute(0,1,3,2)
        w = (v2t_mask+t2v_mask)*mask
        r = w.sum(dim=-1)/w.sum(dim=[-2,-1]).unsqueeze(dim=-1)
        c = w.sum(dim=-2)/w.sum(dim=[-2,-1]).unsqueeze(dim=-1)
        return r, c
    
    def forward(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        return self.aggr(img_cls, imgs, cap_cls, caps, img_lens, cap_lens,)

    def forward_cam(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps

        imgs = imgs / imgs.norm(dim=-1, keepdim=True)
        caps = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs, caps)
        mask = get_fgmask(img_lens,cap_lens)

        v2t_attn = fg_sims.masked_fill(mask == 0, -1e3)/self.lamb
        v2t_attn = torch.softmax(v2t_attn,dim=-1)
        v2t = torch.einsum("itkl,itkl,itkl->itk",v2t_attn,fg_sims,mask)

        t2v_attn = fg_sims.masked_fill(mask == 0, -1e3)/self.lamb
        t2v_attn = torch.softmax(t2v_attn,dim=-2)
        t2v = torch.einsum("itkl,itkl,itkl->itl",t2v_attn,fg_sims,mask)

        r,c = self._ave_init(mask)
        v2t = torch.einsum("itk,itk->it",v2t,r)
        t2v = torch.einsum("itl,itl->it",t2v,c)
        sims = (t2v+v2t)/2
        return sims

    def forward_hichanv2(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        # sims_glo = torch.einsum("ikd,td->itk", imgs, cap_cls).max(dim=-1)[0]
        sims_glo = torch.einsum("id,td->it", img_cls, cap_cls)
        sims = self.forward_chan(img_cls, imgs, cap_cls, caps, img_lens, cap_lens)
        return sims
        # if self.training:
        #     return torch.stack([sims_glo, sims], dim=0)
        # else:
        #     return (sims_glo + self.lamb*sims)/2.0
    

    def forward_hichan(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps
        img_mask = get_mask(img_lens)
        cap_mask = get_mask(cap_lens)

        ## action level
        # imgs_a,_ = self.img_mhsa_0(imgs, img_mask.squeeze())
        # caps_a,_ = self.cap_mhsa_0(caps, cap_mask.squeeze())
        # imgs_e,_ = self.img_mhsa_1(imgs_a)
        # caps_e,_ = self.cap_mhsa_1(caps_a)
        # imgs_all = torch.concat([imgs_e, imgs_a, imgs], dim=1)
        # caps_all = torch.concat([caps_e, caps_a, caps], dim=1)
        # img_lens = img_lens + imgs_a.size(1) + imgs_e.size(1)
        # cap_lens = cap_lens + caps_a.size(1) + caps_e.size(1)
        img_g = imgs.masked_fill(img_mask==0,0).sum(dim=1)/img_mask.sum(dim=1)
        cap_g = caps.masked_fill(cap_mask==0,0).sum(dim=1)/cap_mask.sum(dim=1)
        imgs = torch.concat([img_g.unsqueeze(dim=1), imgs], dim=1)
        caps = torch.concat([cap_g.unsqueeze(dim=1), caps], dim=1)
        imgs = imgs / imgs.norm(dim=-1, keepdim=True)
        caps = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs, caps)
        img_lens = img_lens + 1
        cap_lens = cap_lens + 1

        mask = get_fgmask(img_lens, cap_lens)
        fg_sims = fg_sims.masked_fill(mask == 0, -1)

        r,c = self._ave_init(mask)

        ## entity level
        v2t = torch.einsum("itk,itk->it",fg_sims.max(dim=-1)[0][:,:,1:],r[:,:,1:])
        t2v = torch.einsum("itl,itl->it",fg_sims.max(dim=-2)[0][:,:,1:],c[:,:,1:])
        sims_chan = (t2v+v2t)/2

        return sims_chan
    ## 
    def forward_chan(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps
        img_mask = get_mask(img_lens)
        cap_mask = get_mask(cap_lens)

        imgs_ = imgs / imgs.norm(dim=-1, keepdim=True)
        caps_ = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs_, caps_)
        mask = get_fgmask(img_lens,cap_lens)
        fg_sims = fg_sims.masked_fill(mask == 0, -1)

        r,c = self._ave_init(mask)
        v2t = torch.einsum("itk,itk->it",fg_sims.max(dim=-1)[0],r)
        t2v = torch.einsum("itl,itl->it",fg_sims.max(dim=-2)[0],c)
        sims = (t2v+v2t)/2
        return sims

    def forward_pchan(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:] + self.eps
        caps = caps[:,:max_w,:] + self.eps

        imgs = torch.concat([self.v_ctx.unsqueeze(dim=0).repeat(bi,1,1), imgs], dim=1)
        caps = torch.concat([self.t_ctx.unsqueeze(dim=0).repeat(bt,1,1), caps], dim=1)
                
        # imgs = torch.concat([imgs.max(dim=1,keepdim=True)[0], imgs], dim=1)
        # caps = torch.concat([caps.max(dim=1,keepdim=True)[0], caps], dim=1)
        
        imgs = imgs / imgs.norm(dim=-1, keepdim=True)
        caps = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs, caps)
        img_lens = img_lens + self.nctx
        cap_lens = cap_lens + self.nctx
        mask = get_fgmask(img_lens,cap_lens)
        r,c = self._ave_init(mask)
        v2t = torch.einsum("itk,itk->it",fg_sims.max(dim=-1)[0],r)
        t2v = torch.einsum("itl,itl->it",fg_sims.max(dim=-2)[0],c)
        sims = (t2v+v2t)/2
        return sims
    
    def forward_bimax(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps
        img_mask = get_mask(img_lens)
        cap_mask = get_mask(cap_lens)

        imgs_ = imgs / imgs.norm(dim=-1, keepdim=True)
        caps_ = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs_, caps_)
        mask = get_fgmask(img_lens,cap_lens)
        fg_sims = fg_sims.masked_fill(mask == 0, -1)

        # bimask = self.bimask(fg_sims, mask)
        bimask = self.simask(fg_sims, mask)
        sims = (fg_sims*bimask).sum(dim=[-2,-1]) 
        return sims
    
    @torch.no_grad()
    def bimask(self, sims, mask):
        r,c = self._ave_init(mask)
        # mask = torch.einsum("itk,itl->itkl",r,c)
        # r_mask = sims == sims.max(dim=-1,keepdim=True)[0]
        # c_mask = sims == sims.max(dim=-2,keepdim=True)[0]
        # bimask = mask*(r_mask+c_mask)/2
        # return bimask

        max_inds = sims.argmax(dim=-1,keepdim=True)
        r_mask = torch.zeros_like(sims).to(sims.device)
        # r_mask[:,:,torch.arange(sims.size(-2)),max_inds]=1
        r_mask = r_mask.scatter_(-1,max_inds,1)
        r_mask = torch.einsum("itkl,itk->itkl",r_mask,r)

        max_inds = sims.argmax(dim=-2,keepdim=True)
        c_mask = torch.zeros_like(sims).to(sims.device)
        # c_mask[:,:,max_inds,torch.arange(sims.size(-1))]=1
        c_mask = c_mask.scatter_(-2,max_inds,1)
        c_mask = torch.einsum("itkl,itl->itkl",c_mask,c)

        bimask = (r_mask+c_mask)/2
        return bimask

    @torch.no_grad()
    def simask(self, sims, mask):
        r,c = self._ave_init(mask)
        if True:
            max_inds = sims.argmax(dim=-1,keepdim=True)
            r_mask = torch.zeros_like(sims).to(sims.device)
            # r_mask[:,:,torch.arange(sims.size(-2)),max_inds]=1
            r_mask = r_mask.scatter_(-1,max_inds,1)
            r_mask = torch.einsum("itkl,itk->itkl",r_mask,r)
            return r_mask
        else:
            max_inds = sims.argmax(dim=-2,keepdim=True)
            c_mask = torch.zeros_like(sims).to(sims.device)
            # c_mask[:,:,max_inds,torch.arange(sims.size(-1))]=1
            c_mask = c_mask.scatter_(-2,max_inds,1)
            c_mask = torch.einsum("itkl,itl->itkl",c_mask,c)
            return c_mask

    @torch.no_grad()
    def softmax2d(self, sims, mask):
        sims = sims.masked_fill(mask==0, -INF)
        tau = 0.01 
        mask = torch.exp(sims / tau)  
        mask = mask / (mask.sum(dim=[-2,-1], keepdim=True))
        return mask

    def forward_mhsa(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps
        img_mask = get_mask(img_lens)
        cap_mask = get_mask(cap_lens)

        imgs_ = imgs / imgs.norm(dim=-1, keepdim=True)
        caps_ = caps / caps.norm(dim=-1, keepdim=True)
        fg_sims = get_fgsims(imgs_, caps_)
        mask = get_fgmask(img_lens,cap_lens)

        sims = torch.zeros(3, bi, bt).to(device=caps.device)
        # entity level
        fg_sims = fg_sims.masked_fill(mask == 0, -1)
        r,c = self._ave_init(mask)
        v2t = torch.einsum("itk,itk->it",fg_sims.max(dim=-1)[0],r)
        t2v = torch.einsum("itl,itl->it",fg_sims.max(dim=-2)[0],c)
        sims[0] = (t2v+v2t)/2

        # action level
        imgs,_ = self.img_mhsa_0(imgs, img_mask.squeeze())
        caps,_ = self.cap_mhsa_0(caps, cap_mask.squeeze())
        imgs_ = imgs / (imgs.norm(dim=-1, keepdim=True)+self.eps)
        caps_ = caps / (caps.norm(dim=-1, keepdim=True)+self.eps)
        fg_sims = get_fgsims(imgs_, caps_)
        mask = get_fgmask(torch.zeros(imgs_.size(0))+imgs_.size(1),
                    torch.zeros(caps_.size(0))+caps_.size(1)).to(imgs_.device)
        fg_sims = fg_sims.masked_fill(mask == 0, -1)
        r,c = self._ave_init(mask)
        v2t = torch.einsum("itk,itk->it",fg_sims.max(dim=-1)[0],r)
        t2v = torch.einsum("itl,itl->it",fg_sims.max(dim=-2)[0],c)
        sims[1] = (t2v+v2t)/2

        # event level
        imgs,_ = self.img_mhsa_1(imgs)
        caps,_ = self.cap_mhsa_1(caps)

        imgs_ = imgs / (imgs.norm(dim=-1, keepdim=True)+self.eps)
        caps_ = caps / (caps.norm(dim=-1, keepdim=True)+self.eps)
        fg_sims = get_fgsims(imgs_, caps_)
        mask = get_fgmask(torch.zeros(imgs_.size(0))+imgs_.size(1),
                    torch.zeros(caps_.size(0))+caps_.size(1)).to(imgs_.device)
        fg_sims = fg_sims.masked_fill(mask == 0, -1)
        r,c = self._ave_init(mask)
        v2t = torch.einsum("itk,itk->it",fg_sims.max(dim=-1)[0],r)
        t2v = torch.einsum("itl,itl->it",fg_sims.max(dim=-2)[0],c)
        sims[2] = (t2v+v2t)/2
        return sims

    def forward_cluster(self, img_cls, imgs, cap_cls, caps, img_lens, cap_lens,):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]+self.eps
        caps = caps[:,:max_w,:]+self.eps
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
        fg_sims = fg_sims.masked_fill(mask == 0, -1)
        r,c = self._ave_init(mask)
        v2t = torch.einsum("itk,itk->it",fg_sims.max(dim=-1)[0],r)
        t2v = torch.einsum("itl,itl->it",fg_sims.max(dim=-2)[0],c)
        sims[0] = (t2v+v2t)/2

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

        fg_sims = fg_sims.masked_fill(mask == 0, -1)
        v2t = torch.einsum("itk,itk->it",fg_sims.max(dim=-1)[0],r)
        t2v = torch.einsum("itl,itl->it",fg_sims.max(dim=-2)[0],c)
        sims[1] = (t2v+v2t)/2

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

        fg_sims = fg_sims.masked_fill(mask == 0, -1)
        v2t = torch.einsum("itk,itk->it",fg_sims.max(dim=-1)[0],r)
        t2v = torch.einsum("itl,itl->it",fg_sims.max(dim=-2)[0],c)
        sims[2] = (t2v+v2t)/2
        return sims

if __name__=="__main__":
    batch_size = 128
    imgs = torch.rand([batch_size,36,1024])
    caps = torch.rand([batch_size,51,1024])
    img_lens = torch.randint(20, 37, [batch_size])
    cap_lens = torch.randint(36, 52, [batch_size])

    imgs_ = imgs / imgs.norm(dim=-1, keepdim=True)
    caps_ = caps / caps.norm(dim=-1, keepdim=True)
    fg_sims = get_fgsims(imgs_, caps_)
    mask = get_fgmask(img_lens,cap_lens)
    fg_sims = fg_sims.masked_fill(mask == 0, -1)
    v2t_index = fg_sims.argmax(dim=-1, keepdim=True)
    t2v_index = fg_sims.argmax(dim=-2, keepdim=True)
    v2t_mask = v2t_index.repeat(1,1,1,fg_sims.shape[-1])==torch.arange(fg_sims.shape[-1]).expand_as(fg_sims)
    t2v_mask = t2v_index.repeat(1,1,fg_sims.shape[-2],1)==torch.arange(fg_sims.shape[-2]).expand_as(fg_sims.permute(0,1,3,2)).permute(0,1,3,2)
    w = 1/(v2t_mask+t2v_mask).sum(dim=[-2,-1])

    pass
