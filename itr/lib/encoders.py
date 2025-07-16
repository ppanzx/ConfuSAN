"""VSE modules"""
import os
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace

import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertModel

from lib.modules.resnet import ResnetFeatureExtractor
from lib.modules.utils import l2norm, MLP, SelfAttention, Transformer
from lib.modules.utils import get_mask,random_masking
from lib.modules.module_clip import Transformer as TransformerClip
from lib.modules.gcn import Rs_GCN as gcn
from clip import clip

import logging
logger = logging.getLogger(__name__)

def load_clip_to_cpu(clip_model_name):
    url = clip._MODELS[clip_model_name]
    model_path = clip._download(url)
    # ckpt_path = "~/project/pzx/dataset/hub/clip"
    # model_path = osp.expanduser(osp.join(ckpt_path,osp.basename(url)))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def get_text_encoder(vocab_size, embed_size, word_dim, num_layers, text_enc_type="bigru", 
                    use_bi_gru=True, no_txtnorm=False, **args):
    """A wrapper to text encoders."""
    if text_enc_type == "bigru":
        txt_enc = EncoderTextBigru(vocab_size, embed_size, word_dim, num_layers, use_bi_gru=use_bi_gru, no_txtnorm=no_txtnorm, **args)
    elif text_enc_type == "bert":
        txt_enc = EncoderTextBert(embed_size, no_txtnorm=no_txtnorm)
    elif text_enc_type == "clip":
        clip_model = load_clip_to_cpu("ViT-B/32")
        txt_enc = EncoderTextCLIP(clip_model, embed_size=embed_size, no_txtnorm=no_txtnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(text_enc_type))
    return txt_enc


def get_image_encoder(img_dim, embed_size, precomp_enc_type='basic', backbone_source=None, 
                        backbone_path=None, no_imgnorm=False, visual_mask_ratio=0.2, **args):
    """A wrapper to image encoders."""
    if precomp_enc_type == 'backbone':
        backbone_cnn = ResnetFeatureExtractor(backbone_source, backbone_path, fixed_blocks=2)
        img_enc = EncoderImageFull(backbone_cnn, img_dim, embed_size, precomp_enc_type, no_imgnorm, visual_mask_ratio, **args)
    elif precomp_enc_type == 'clip':
        clip_model = load_clip_to_cpu("ViT-B/32")
        img_enc = EncoderImageCLIP(clip_model, embed_size=embed_size, no_imgnorm=no_imgnorm, mask_ratio=visual_mask_ratio)
    else:
        img_enc = EncoderImagePrecomp(img_dim, embed_size, precomp_enc_type, no_imgnorm, visual_mask_ratio, **args)
    return img_enc

class EncoderImageFull(nn.Module):
    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, mask_ratio=0.2):
        super(EncoderImageFull, self).__init__()
        self.backbone = backbone_cnn
        self.backbone_freezed = False
        self.image_encoder = EncoderImagePrecomp(img_dim, embed_size, precomp_enc_type, no_imgnorm, mask_ratio)

    def forward(self, images, feat_lengths):
        """Extract image feature vectors."""
        base_features = self.backbone(images)

        cls, base_features, feat_lengths = self.image_encoder(base_features, feat_lengths)

        return cls, base_features, feat_lengths

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))


class EncoderImagePrecomp(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, mask_ratio=0.2, **args):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.mask_ratio = mask_ratio
        self.precomp_enc_type = precomp_enc_type
        self.fc = nn.Linear(img_dim, embed_size)

        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": int(args["opt"].alpha),
            "vocab_size": 512,
            "soft_t": 0.07,
        })

        if precomp_enc_type=="basic":
            self.feedforward = nn.Identity()
        elif precomp_enc_type == "mlp":
            self.feedforward = MLP(embed_size, embed_size // 2, embed_size, 2)
        elif precomp_enc_type=="selfattention":
            self.feedforward = SelfAttention(embed_size)
        elif self.precomp_enc_type == "seqTransf" or precomp_enc_type=="backbone":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            self.transformerClip = TransformerClip(width=embed_size,
                                                    layers=cross_config.num_hidden_layers,
                                                    heads=embed_size)
        elif self.precomp_enc_type == "seqLSTM":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                        batch_first=True, bidirectional=False, num_layers=1)
        elif precomp_enc_type=="gcn":
            self.gcns = nn.ModuleList([gcn(cross_config.hidden_size, cross_config.hidden_size) \
                                       for i in range(cross_config.num_hidden_layers)])
            self.gru_visual = nn.GRU(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                        batch_first=True, bidirectional=True, num_layers=1)
        else:
            raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def agg_video_feat(self, video_feat, video_mask, agg_module):
        video_feat = video_feat.contiguous()
        if agg_module == "basic":
            pass
        elif agg_module == "mlp":
            video_feat_original = video_feat
            video_feat = self.feedforward(video_feat)
            video_feat = video_feat + video_feat_original
        elif agg_module == "seqLSTM":
            # Sequential type: LSTM
            video_feat_original = video_feat
            video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(),
                                              batch_first=True, enforce_sorted=False)
            video_feat, _ = self.lstm_visual(video_feat)
            if self.training: self.lstm_visual.flatten_parameters()
            video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
            video_feat = torch.cat(
                (video_feat, video_feat_original[:, video_feat.size(1):, ...].contiguous()), dim=1)
            video_feat = video_feat + video_feat_original
        elif agg_module == "seqTransf" or agg_module=="backbone":
            # Sequential type: Transformer Encoder
            video_feat_original = video_feat
            seq_length = video_feat.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            video_feat = video_feat + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            video_feat = video_feat.permute(1, 0, 2)  # NLD -> LND
            # video_feat = self.transformerClip(video_feat, extended_video_mask)
            video_feat = self.transformerClip(video_feat, None)
            video_feat = video_feat.permute(1, 0, 2)  # LND -> NLD
            video_feat = video_feat + video_feat_original
        elif agg_module == "gcn":
            ## GCN
            video_feat = video_feat.permute(0, 2, 1)
            for module in self.gcns:
                # video_feat = l2norm(video_feat, dim=1)
                video_feat = module(video_feat)
            video_feat = video_feat.permute(0, 2, 1)

            ## seq2seq
            # video_feat = l2norm(video_feat, dim=-1)
            # video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(),
            #                                   batch_first=True, enforce_sorted=False)
            # video_feat, hidden = self.gru_visual(video_feat)
            # if self.training: self.gru_visual.flatten_parameters()
            # video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
            # video_feat = (video_feat[:, :, :video_feat.size(2) // 2] + video_feat[:, :, video_feat.size(2) // 2:]) / 2
        return video_feat

    def forward(self, base_features, feat_lengths):
        """Extract image feature vectors."""
        if self.training and self.mask_ratio>0:
            # Size Augmentation during training, randomly mask features
            base_features, mask, _ = random_masking(base_features, self.mask_ratio)
            feat_lengths = (mask==0).sum(dim=-1).to(dtype=torch.int64)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)
       
        mask = get_mask(feat_lengths)
        # cls = base_features.masked_fill(mask==0,0).sum(dim=1)/mask.sum(dim=1)
        # base_features = torch.concat([cls.unsqueeze(dim=1), base_features], dim=1)

        features = self.fc(base_features)

        features = self.agg_video_feat(features, mask.squeeze(), self.precomp_enc_type)
        cls = features.masked_fill(mask==0,0).sum(dim=1)/mask.sum(dim=1)
        # cls = features[:,0]

        if not self.no_imgnorm:
            cls = l2norm(cls, dim=-1)
            features = l2norm(features, dim=-1)

        return cls, features, feat_lengths

# Visual Model with VIT in CLIP
class EncoderImageCLIP(nn.Module):
    """
        Note: Only support for Visual Transformer
    """
    def __init__(self, clip_model, embed_size=1024, no_imgnorm=False, mask_ratio=0, fixed_blocks=2, **args):
        super(EncoderImageCLIP, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.mask_ratio = mask_ratio
        self.fixed_blocks = fixed_blocks
        # self.dtype = clip_model.dtype
        self.dtype = torch.float32

        self.clip = clip_model.visual.to(self.dtype)

        if embed_size!=clip_model.visual.output_dim:
            width = clip_model.visual.proj.shape[0]
            scale = width ** -0.5
            self.proj = nn.Parameter(scale * torch.randn(width, embed_size)).to(self.dtype)
        else:
            self.proj = nn.Parameter(clip_model.visual.proj.data.to(dtype=self.dtype))
            logger.info('Directly optimize the projection of original visual CLIP.')
            
        self.freeze_base()

    def forward(self, x: torch.Tensor, feat_lengths):
        ## patchify
        x = self.clip.conv1(x.type(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.clip.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], 
                        dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.positional_embedding[:x.shape[1]]
        x = self.clip.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_post(x)

        # random masking
        if self.training and self.mask_ratio>0:
            x, mask, _ = random_masking(x, self.mask_ratio)
            feat_lengths = (mask==0).sum(dim=-1).to(dtype=torch.int64)
        else:
            feat_lengths = torch.zeros(x.size(0)).to(x.device)
            feat_lengths[:] = x.size(1)

        hidden = x @ self.proj

        if not self.no_imgnorm:
            hidden = l2norm(hidden, dim=-1)

        x = hidden[:, 0, :]

        return x, hidden, feat_lengths

    def unfreeze_base(self, ):
        assert (0 <= self.fixed_blocks < 4)
        if self.fixed_blocks == 3:
            for p in self.clip.transformer.resblocks[10:12].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[8:10].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[6:8].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[4:6].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[2:4].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[0:2].parameters(): p.requires_grad = False
        elif self.fixed_blocks == 2:
            for p in self.clip.transformer.resblocks[10:12].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[8:10].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[6:8].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[4:6].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[2:4].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[0:2].parameters(): p.requires_grad = False
        elif self.fixed_blocks == 1:
            for p in self.clip.transformer.resblocks[10:12].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[8:10].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[6:8].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[4:6].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[2:4].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[0:2].parameters(): p.requires_grad = False
        elif self.fixed_blocks == 0:
            for p in self.clip.transformer.resblocks[10:12].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[8:10].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[6:8].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[4:6].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[2:4].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[0:2].parameters(): p.requires_grad = True
            for p in self.parameters():
                p.requires_grad = True

        logger.info('CLIP backbone now has fixed blocks {}'.format(self.fixed_blocks))

    def freeze_base(self):
        for p in self.parameters():
            p.requires_grad = False
        # for p in self.clip.ln_post.parameters(): p.requires_grad = True
        self.proj.requires_grad = True

    def set_fixed_blocks(self, fixed_blocks):
        self.fixed_blocks = fixed_blocks

    def get_fixed_blocks(self):
        return self.fixed_blocks

    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
        # for p in self.clip.ln_post.parameters(): p.requires_grad = True
        self.proj.requires_grad = True
        logger.info('Visual CLIP freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.clip.transformer.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.set_fixed_blocks(fixed_blocks)
        self.unfreeze_base()
        logger.info('Visual CLIP unfreezed, fixed blocks {}'.format(self.get_fixed_blocks()))


# Language Model with BiGRU
class EncoderTextBigru(nn.Module):
    def __init__(self, vocab_size, embed_size, word_dim, num_layers, use_bi_gru=True, no_txtnorm=False, **args):
        super(EncoderTextBigru, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        hidden_size = embed_size
        self.rnn = nn.GRU(word_dim, hidden_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        self.fc = nn.Linear(hidden_size, embed_size)
        self.init_weights(wemb_type=args["wemb_type"],word2idx=args["word2idx"],word_dim=word_dim)

    def init_weights(self, wemb_type="glove", word2idx=None, word_dim=300, cache_dir="~/dataset/dependency/ckpt/"):
        if wemb_type is None or word2idx is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            cache_dir = os.path.expanduser(cache_dir+wemb_type)
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText(cache=cache_dir)
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache=cache_dir)
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            ##
            self.embed.requires_grad = False
            logger.info('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths, return_hidden=True):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)

        self.rnn.flatten_parameters()
        packed = pack_padded_sequence(x_emb, lengths.cpu(), batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2
        
        # mask = get_mask(lengths)
        # cls = cap_emb.masked_fill(mask==0,0).sum(dim=1)/mask.sum(dim=1)
        I = lengths.expand(self.embed_size, 1, -1).permute(2, 1, 0) - 1
        cls = torch.gather(cap_emb, 1, I.long()).squeeze(1)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cls = l2norm(cls, dim=-1)
            cap_emb = l2norm(cap_emb, dim=-1)

        return cls, cap_emb


# Language Model with BERT
class EncoderTextBert(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderTextBert, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        root = os.path.expanduser("~/dataset/dependency/ckpt/transformers/bert-base")
        self.bert = BertModel.from_pretrained(config=root,pretrained_model_name_or_path=root)
        
        self.linear = nn.Linear(768, embed_size)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D

        cap_emb = self.linear(bert_emb)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb[:,0], cap_emb

# Language Model with Transformers in CLIP
class EncoderTextCLIP(nn.Module):
    def __init__(self, clip_model, embed_size=1024, no_txtnorm=False, fixed_blocks=2, mask_ratio=0):
        super(EncoderTextCLIP, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.fixed_blocks = fixed_blocks
        # self.dtype = clip_model.dtype
        self.mask_ratio = mask_ratio
        self.dtype = torch.float32
        
        self.clip = nn.ModuleList(modules=None)
        self.clip.transformer = clip_model.transformer.to(self.dtype)
        self.clip.token_embedding = clip_model.token_embedding.to(self.dtype)
        self.clip.positional_embedding = clip_model.positional_embedding.to(self.dtype)
        self.clip.ln_final = clip_model.ln_final.to(self.dtype)

        transformer_width = clip_model.transformer.width
        output_dim = clip_model.text_projection.shape[0]
        if embed_size!=output_dim:
            scale = transformer_width ** -0.5
            self.proj = nn.Parameter(scale * torch.randn(transformer_width, embed_size)).to(self.dtype)
        else:
            self.proj = nn.Parameter(clip_model.text_projection.data.to(dtype=self.dtype))
            logger.info('Directly optimize the projection of original textual CLIP.')
        self.freeze_base()

    def forward(self, text, lengths):
        x = self.clip.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        pos_emd = self.clip.positional_embedding[:x.size(1), :]
        x = x + pos_emd
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x) 

        # random masking
        if self.training and self.mask_ratio>0:
            x, mask, _ = random_masking(x, self.mask_ratio)
            feat_lengths = (mask==0).sum(dim=-1).to(dtype=torch.int64)
        else:
            feat_lengths = torch.zeros(x.size(0)).to(x.device)
            feat_lengths[:] = x.size(1)

        hidden = x @ self.proj

        if not self.no_txtnorm:
            hidden = l2norm(hidden, dim=-1)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]

        return x, hidden

    def unfreeze_base(self):
        assert (0 <= self.fixed_blocks < 4)
        if self.fixed_blocks == 3:
            for p in self.clip.transformer.resblocks[10:12].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[8:10].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[6:8].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[4:6].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[2:4].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[0:2].parameters(): p.requires_grad = False
        elif self.fixed_blocks == 2:
            for p in self.clip.transformer.resblocks[10:12].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[8:10].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[6:8].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[4:6].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[2:4].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[0:2].parameters(): p.requires_grad = False
        elif self.fixed_blocks == 1:
            for p in self.clip.transformer.resblocks[10:12].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[8:10].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[6:8].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[4:6].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[2:4].parameters(): p.requires_grad = False
            for p in self.clip.transformer.resblocks[0:2].parameters(): p.requires_grad = False
        elif self.fixed_blocks == 0:
            for p in self.clip.transformer.resblocks[10:12].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[8:10].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[6:8].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[4:6].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[2:4].parameters(): p.requires_grad = True
            for p in self.clip.transformer.resblocks[0:2].parameters(): p.requires_grad = True
            for p in self.parameters():
                p.requires_grad = True

        logger.info('CLIP backbone now has fixed blocks {}'.format(self.fixed_blocks))

    def freeze_base(self):
        for p in self.parameters():
            p.requires_grad = False
        # for p in self.clip.ln_final.parameters(): p.requires_grad = True
        self.proj.requires_grad = True

    def set_fixed_blocks(self, fixed_blocks):
        self.fixed_blocks = fixed_blocks

    def get_fixed_blocks(self):
        return self.fixed_blocks

    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
        # for p in self.clip.ln_final.parameters(): p.requires_grad = True
        self.proj.requires_grad = True
        logger.info('Textual CLIP freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.clip.transformer.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.set_fixed_blocks(fixed_blocks)
        self.unfreeze_base()
        logger.info('Textual CLIP unfreezed, fixed blocks {}'.format(self.get_fixed_blocks()))

# experiments on aggregation types
from lib.aggr.pooling import aveEncoders
from lib.aggr.gpo import gpoEncoders
from lib.aggr.pcme import PEM
from lib.aggr.coding import get_coding,get_pooling
from lib.aggr.ot import Wasserstain
# from lib.aggr.cam import get_coding # cover the coding in conding.py
from lib.aggr.select import Select
from lib.aggr.nmf import NMF
from lib.aggr.hausdorff import Hausdorff
from lib.aggr.ctf import CTF
from lib.aggr.hot import HOT
from lib.aggr.hichan import HiCHAN
from lib.aggr.hcam import HCAM
from lib.aggr.sot import SparseOT
class SimsEncoder(nn.Module):
    def __init__(self, coding_type, pooling_type, **args):
        super(SimsEncoder, self).__init__()
        self.opt = args["opt"]
        self.aggr_type=args["opt"].aggr_type

        if self.aggr_type=="ave":
            ## 1. average pooling
            self.ave = aveEncoders()
        elif self.aggr_type=="gpo":
            ## 2. gpo pooling
            self.gpo = gpoEncoders(32,32)
        elif self.aggr_type=="pem":
            self.pem = PEM()
        elif self.aggr_type=="coding":
            ## 3. coding
            self.coding = get_coding(coding_type, opt=self.opt)
            self.pooling = get_pooling(pooling_type, opt=self.opt)
        elif self.aggr_type=="ot":
            ## 4. optimal transport
            self.sinkhorn = Wasserstain(lamb=self.opt.alpha, iters=int(self.opt.belta), _init=self.opt.KNN)
        elif self.aggr_type=="select":
            ## 5. token flow
            self.select = Select(self.opt.KNN)
        elif self.aggr_type=="nmf":
            self.nmf = NMF(iters=int(self.opt.belta))
        elif self.aggr_type=="hausdorff":
            self.hsd = Hausdorff()
        elif self.aggr_type=="ctf":
            self.ctf = CTF(embed_dim=self.opt.embed_size)
        elif self.aggr_type=="hot":
            self.hot = HOT(lamb=self.opt.alpha, iters=int(self.opt.belta), _init=coding_type)
        elif self.aggr_type=="hichan":
            self.hichan = HiCHAN(lamb=self.opt.alpha, iters=int(self.opt.belta), _init=coding_type)
        elif self.aggr_type=="hcam":
            self.hcam = HCAM(lamb=self.opt.alpha, iters=int(self.opt.belta), _init=self.opt.KNN)
        elif self.aggr_type=="sot":
            self.sot = SparseOT(lamb=self.opt.alpha, iters=int(self.opt.belta))

    def forward(self, img_cls, img_embs, cap_cls, cap_embs, img_lens, cap_lens,):
        if self.aggr_type=="ave":
            ## 1. average pooling
            sims = self.ave(img_cls, img_embs, cap_cls, cap_embs, img_lens, cap_lens,)
        elif self.aggr_type=="gpo":
            ## 2. gpo pooling
            sims = self.gpo(img_cls, img_embs, cap_cls, cap_embs, img_lens, cap_lens,)
        elif self.aggr_type=="pem":
            sims = self.pem(img_cls, img_embs, cap_cls, cap_embs, img_lens, cap_lens,)
        elif self.aggr_type=="coding":
            ## 3. coding
            sims = self.coding(img_cls, img_embs, cap_cls, cap_embs, img_lens, cap_lens,)
            sims = self.pooling(sims)
        elif self.aggr_type=="ot":
            # 4. optimal transport
            sims = self.sinkhorn(img_cls, img_embs, cap_cls, cap_embs, img_lens, cap_lens,)
        elif self.aggr_type=="select":
            ## 5. token flow
            sims = self.select(img_cls, img_embs, cap_cls, cap_embs, img_lens, cap_lens,)
        elif self.aggr_type=="cosine":
            sims = img_cls @ cap_cls.t()
        elif self.aggr_type=="nmf":
            sims = self.nmf(img_cls, img_embs, cap_cls, cap_embs, img_lens, cap_lens,)
        elif self.aggr_type=="hausdorff":
            sims = self.hsd(img_cls, img_embs, cap_cls, cap_embs, img_lens, cap_lens,)
        elif self.aggr_type=="ctf":
            sims = self.ctf(img_cls, img_embs, cap_cls, cap_embs, img_lens, cap_lens,)
        elif self.aggr_type=="hot":
            sims = self.hot(img_cls, img_embs, cap_cls, cap_embs, img_lens, cap_lens,)
        elif self.aggr_type=="hichan":
            sims = self.hichan(img_cls, img_embs, cap_cls, cap_embs, img_lens, cap_lens,)
        elif self.aggr_type=="hcam":
            sims = self.hcam(img_cls, img_embs, cap_cls, cap_embs, img_lens, cap_lens,)
        elif self.aggr_type=="sot":
            sims = self.sot(img_cls, img_embs, cap_cls, cap_embs, img_lens, cap_lens,)
        
        return sims