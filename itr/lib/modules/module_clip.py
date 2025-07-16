from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import math
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0) if attn_mask else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]


class CrossEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(CrossEmbeddings, self).__init__()

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, concat_embeddings, concat_type=None):
        batch_size, seq_length = concat_embeddings.size(0), concat_embeddings.size(1)
        # if concat_type is None:
        #     concat_type = torch.zeros(batch_size, concat_type).to(concat_embeddings.device)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=concat_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(concat_embeddings.size(0), -1)

        # token_type_embeddings = self.token_type_embeddings(concat_type)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = concat_embeddings + position_embeddings  # + token_type_embeddings
        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CrossPooler(nn.Module):
    def __init__(self, config):
        super(CrossPooler, self).__init__()
        self.ln_pool = LayerNorm(config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = QuickGELU()

    def forward(self, hidden_states, hidden_mask):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = self.ln_pool(hidden_states)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CrossModel(nn.Module):

    def initialize_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def __init__(self, config):
        super(CrossModel, self).__init__()
        self.config = config

        self.embeddings = CrossEmbeddings(config)

        transformer_width = config.hidden_size
        transformer_layers = config.num_hidden_layers
        transformer_heads = config.num_attention_heads
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, )
        self.pooler = CrossPooler(config)
        self.apply(self.init_weights)

    def build_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0
        extended_attention_mask = extended_attention_mask.expand(-1, attention_mask.size(1), -1)
        return extended_attention_mask

    def forward(self, concat_input, concat_type=None, attention_mask=None, output_all_encoded_layers=True):

        if attention_mask is None:
            attention_mask = torch.ones(concat_input.size(0), concat_input.size(1))
        if concat_type is None:
            concat_type = torch.zeros_like(attention_mask)

        extended_attention_mask = self.build_attention_mask(attention_mask)

        embedding_output = self.embeddings(concat_input, concat_type)
        embedding_output = embedding_output.permute(1, 0, 2)  # NLD -> LND
        embedding_output = self.transformer(embedding_output, extended_attention_mask)
        embedding_output = embedding_output.permute(1, 0, 2)  # LND -> NLD

        pooled_output = self.pooler(embedding_output, hidden_mask=attention_mask)

        return embedding_output, pooled_output

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        num_texts, _ = text_embeds.shape
        # num_texts x embed_dim
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1,2,0)

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)
        return o


class Crossformer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(Crossformer, self).__init__()
        self.embed_dim = embed_dim

        self.cross_attn = MultiHeadedAttention(embed_dim, num_heads)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out


# fine-grained sinkhorn distance
class Wasserstain(nn.Module):
    def __init__(self, iters=100, lamb=1e-2,):
        super(Wasserstain, self).__init__()
        self.eps = 1e-6
        self.iters = int(iters)
        self.lamb = lamb
    
    def _ave_init(self, mask):
        r = mask.sum(dim=[-1])!=0  # Bi x Bt x K
        c = mask.sum(dim=[-2])!=0  # Bi x Bt x L
        r = r*(1/r.sum(dim=-1, keepdim=True))
        c = c*(1/c.sum(dim=-1, keepdim=True))
        return r, c

    @torch.no_grad()
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

    def forward(self, text_embeds, video_embeds, cap_mask, vid_mask, r=None, c=None):
        cap_mask = cap_mask.unsqueeze(dim=-1)
        vid_mask = vid_mask.unsqueeze(dim=-1)
        mask = torch.einsum("tld,vfd->tvlf", cap_mask, vid_mask)
        sims = torch.einsum("tld,vfd->tvlf", text_embeds, video_embeds)
        cost = (1-sims).masked_fill(mask==0, float("inf"))
        if r is None or c is None:
            r,c = self._ave_init(mask)
        otp = self.Sinkhorn_Knopp(cost, r, c)
        otp = otp.masked_fill(mask==0, 0)
        sims = (sims*otp).sum(dim=[-2,-1])
        return sims

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B*N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x

class NMFormer(nn.Module):
    def __init__(self, embed_dim, dropout=0.4):
        super(NMFormer, self).__init__()
        self.embed_dim = embed_dim

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.linear_proj1 = nn.Linear(self.embed_dim, self.embed_dim)
        # self.linear_proj2 = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        sims = torch.einsum("vfd,td->vtf", video_embeds, text_embeds) # Bi x Bt x K x L

        # initialize the attention matrix
        attn_tilde = sims.clone()
        # attn_tilde = torch.rand_like(sims).to(sims.device)

        self.iters = 10
        self.lamb = 15e-2
        self.eps = 1e-6
        margin=-0

        eta = self.lamb/self.iters
        for i in range(self.iters):
            dnm = torch.einsum("vmd,vnd,vtn->vtm", video_embeds, video_embeds, attn_tilde) + self.eps # denominator
            attn_tilde = attn_tilde - eta*(dnm-sims)
            attn_tilde[attn_tilde <= margin] = self.eps
            norm = torch.einsum("vtf,vfd->vtd", attn_tilde, video_embeds).norm(p=2, dim=-1, keepdim=True)
            attn =  attn_tilde/norm # 
            if (sims-dnm).abs().max()<self.eps:
                break
        # selection = sims.permute(1,0,2).max(dim=-1)[0]
        attn_out = torch.einsum("vfd,vtf->vtd", video_embeds, attn)

        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out
    
class PyramidNMF(nn.Module):
    def __init__(self, embed_dim, pyramids=[1024]):
        super(PyramidNMF, self).__init__()
        self.embed_dim = embed_dim

        self.layers = len(pyramids)
        self.visprojs = nn.ModuleList([nn.Sequential(
                nn.Linear(self.embed_dim, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden)) for hidden in pyramids])
        # self.txtprojs = nn.ModuleList([nn.Linear(self.embed_dim, hidden) for hidden in pyramids])

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, text_embeds, video_embeds):
        sims_p = torch.zeros(text_embeds.shape[0],video_embeds.shape[0],
                                self.layers).to(device=text_embeds.device)
        for ind in range(self.layers):
            text_feat = self.visprojs[ind](text_embeds)
            text_feat = text_feat.squeeze(1)  # B x 1 x D -> B x D
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  # B x D
            video_feat = self.visprojs[ind](video_embeds)
            video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
            sims = torch.einsum("vfd,td->vtf", video_feat, text_feat) 

            # initialize the attention matrix
            attn_tilde = sims.clone()
            # attn_tilde = torch.rand_like(sims).to(sims.device)

            self.iters = 5
            self.lamb = 15e-2
            self.eps = 1e-6
            margin=-0

            eta = self.lamb/self.iters
            for i in range(self.iters):
                dnm = torch.einsum("vmd,vnd,vtn->vtm", video_feat, video_feat, attn_tilde) + self.eps # denominator
                attn_tilde = attn_tilde - eta*(dnm-sims)
                attn_tilde[attn_tilde <= margin] = self.eps
                norm = torch.einsum("vtf,vfd->vtd", attn_tilde, video_feat).norm(p=2, dim=-1, keepdim=True)
                attn =  attn_tilde/norm # 
                if (sims-dnm).abs().max()<self.eps:
                    break
            # selection = sims.permute(1,0,2).max(dim=-1)[0]
            sims_p[:,:,ind] = torch.einsum("vtf,vtf->tv", sims, attn)
        return sims_p.max(dim=-1)[0]
    
def sparsemax(input, dim=-1):
    device = input.device
    number_of_logits = input.size(dim)

    # Translate input by max for numerical stability
    input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

    # Sort input in descending order.
    # (NOTE: Can be replaced with linear time selection method described here:
    # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
    zs = torch.sort(input=input, dim=dim, descending=True)[0]
    range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype)
    range = range.expand_as(zs)

    # Determine sparsity of projection
    bound = 1 + range * zs
    cumulative_sum_zs = torch.cumsum(zs, dim)
    is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
    k = torch.max(is_gt * range, dim, keepdim=True)[0]

    # Compute threshold function
    zs_sparse = is_gt * zs

    # Compute taus
    taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
    taus = taus.expand_as(input)

    # Sparsemax
    output = torch.max(torch.zeros_like(input), input - taus)
    return output