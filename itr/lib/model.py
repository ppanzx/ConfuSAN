"""CHAN model"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast

from lib.encoders import get_image_encoder, get_text_encoder, SimsEncoder
from lib.loss import get_criterion

import logging

logger = logging.getLogger(__name__)

if torch.cuda.device_count()==1:
    img_device="cuda:0"
    txt_device="cuda:0"
    sim_device="cuda:0"
    loss_device="cuda:0"
    device_ids = [0]
elif torch.cuda.device_count()==2:
    img_device="cuda:0"
    txt_device="cuda:0"
    sim_device="cuda:1"
    loss_device="cuda:0"
    device_ids = [0]
elif torch.cuda.device_count()==3:
    img_device="cuda:0"
    txt_device="cuda:0"
    sim_device="cuda:2"
    loss_device="cuda:0"
    device_ids = [0,1]
elif torch.cuda.device_count()==4:
    img_device="cuda:0"
    txt_device="cuda:0"
    sim_device="cuda:3"
    loss_device="cuda:0"
    device_ids = [0,1,2]

    
class Model(object):
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = get_image_encoder(opt.img_dim, opt.embed_size,
                                         precomp_enc_type=opt.precomp_enc_type,
                                         backbone_source=opt.backbone_source,
                                         backbone_path=opt.backbone_path,
                                         no_imgnorm=opt.no_imgnorm,
                                         visual_mask_ratio=opt.visual_mask_ratio,
                                         opt=opt)
        self.txt_enc = get_text_encoder(opt.vocab_size, opt.embed_size, opt.word_dim, opt.num_layers,
                                         text_enc_type=opt.text_enc_type, use_bi_gru=True, 
                                         no_txtnorm=opt.no_txtnorm, wemb_type=opt.wemb_type,
                                         word2idx=opt.word2idx)
        self.sim_enc = SimsEncoder(coding_type=opt.coding_type, pooling_type=opt.pooling_type, opt=opt)

        # Loss and Optimizer
        self.criterion = get_criterion(opt.criterion,opt)
        
        if torch.cuda.is_available():
            self.img_enc.to(device=img_device)
            self.txt_enc.to(device=txt_device)
            self.sim_enc.to(device=sim_device)
            self.criterion.to(device=loss_device)
            cudnn.benchmark = True

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.sim_enc.parameters())
        params += list(self.criterion.parameters())

        self.params = params
        self.opt = opt

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4
        if opt.text_enc_type == 'bigru':
            if opt.precomp_enc_type == 'backbone':
                if self.opt.optim == 'adam':
                    self.optimizer = torch.optim.AdamW([
                        {'params': self.txt_enc.parameters(), 'lr': opt.learning_rate},
                        {'params': self.img_enc.backbone.top.parameters(),
                        'lr': opt.learning_rate * opt.visual_lr_factor, },
                        {'params': self.img_enc.backbone.base.parameters(),
                        'lr': opt.learning_rate * opt.visual_lr_factor, },
                        {'params': self.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                    ], lr=opt.learning_rate, weight_decay=decay_factor)
                else:
                    raise ValueError('Invalid optim option {}'.format(self.opt.optim))
            else:
                if self.opt.optim == 'adam':
                    self.optimizer = torch.optim.AdamW(self.params, lr=opt.learning_rate)
                else:
                    raise ValueError('Invalid optim option {}'.format(self.opt.optim))
        elif opt.text_enc_type == 'bert':
            if opt.precomp_enc_type == 'backbone':
                if self.opt.optim == 'adam':
                    all_text_params = list(self.txt_enc.parameters())
                    bert_params = list(self.txt_enc.bert.parameters())
                    bert_params_ptr = [p.data_ptr() for p in bert_params]
                    text_params_no_bert = list()
                    for p in all_text_params:
                        if p.data_ptr() not in bert_params_ptr:
                            text_params_no_bert.append(p)
                    self.optimizer = torch.optim.AdamW([
                        {'params': text_params_no_bert, 'lr': opt.learning_rate},
                        {'params': bert_params, 'lr': opt.learning_rate * opt.text_lr_factor},
                        {'params': self.img_enc.backbone.top.parameters(),
                        'lr': opt.learning_rate * opt.visual_lr_factor, },
                        {'params': self.img_enc.backbone.base.parameters(),
                        'lr': opt.learning_rate * opt.visual_lr_factor, },
                        {'params': self.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                    ], lr=opt.learning_rate, weight_decay=decay_factor)
                elif self.opt.optim == 'sgd':
                    self.optimizer = torch.optim.SGD([
                        {'params': self.txt_enc.parameters(), 'lr': opt.learning_rate},
                        {'params': self.img_enc.backbone.parameters(), 'lr': opt.learning_rate * opt.visual_lr_factor,
                        'weight_decay': decay_factor},
                        {'params': self.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                    ], lr=opt.learning_rate, momentum=0.9, nesterov=True)
                else:
                    raise ValueError('Invalid optim option {}'.format(self.opt.optim))
            elif opt.precomp_enc_type == 'clip':
                all_text_params = list(self.txt_enc.parameters())
                bert_params = list(self.txt_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)

                all_visual_params = list(self.img_enc.parameters())
                visual_params_clip = list(self.img_enc.clip.parameters())
                visual_clip_params_ptr = [p.data_ptr() for p in visual_params_clip]
                visual_params_no_clip = list()
                for p in all_visual_params:
                    if p.data_ptr() not in visual_clip_params_ptr:
                        visual_params_no_clip.append(p)

                ## optimizer
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate,},
                    {'params': visual_params_no_clip, 'lr': opt.learning_rate,},
                    {'params': bert_params, 'lr': opt.learning_rate * opt.text_lr_factor,},
                    {'params': visual_params_clip,'lr': opt.learning_rate * opt.visual_lr_factor,},
                ], lr=opt.learning_rate, weight_decay=decay_factor)
            else:
                if self.opt.optim == 'adam':
                    all_text_params = list(self.txt_enc.parameters())
                    bert_params = list(self.txt_enc.bert.parameters())
                    bert_params_ptr = [p.data_ptr() for p in bert_params]
                    text_params_no_bert = list()
                    for p in all_text_params:
                        if p.data_ptr() not in bert_params_ptr:
                            text_params_no_bert.append(p)
                    self.optimizer = torch.optim.AdamW([
                        {'params': text_params_no_bert, 'lr': opt.learning_rate},
                        {'params': bert_params, 'lr': opt.learning_rate * opt.text_lr_factor},
                        {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
                    ],
                        lr=opt.learning_rate, weight_decay=decay_factor)
                elif self.opt.optim == 'sgd':
                    self.optimizer = torch.optim.SGD(self.params, lr=opt.learning_rate, momentum=0.9)
                else:
                    raise ValueError('Invalid optim option {}'.format(self.opt.optim))
        elif opt.text_enc_type == 'clip':
            # decay_factor = 0.2
            all_text_params = list(self.txt_enc.parameters())
            text_params_clip = list(self.txt_enc.clip.parameters())
            text_clip_params_ptr = [p.data_ptr() for p in text_params_clip]
            text_params_no_clip = list()
            for p in all_text_params:
                if p.data_ptr() not in text_clip_params_ptr:
                    text_params_no_clip.append(p)

            all_visual_params = list(self.img_enc.parameters())
            visual_params_clip = list(self.img_enc.clip.parameters())
            visual_clip_params_ptr = [p.data_ptr() for p in visual_params_clip]
            visual_params_no_clip = list()
            for p in all_visual_params:
                if p.data_ptr() not in visual_clip_params_ptr:
                    visual_params_no_clip.append(p)

            ## optimizer
            self.optimizer = torch.optim.AdamW([
                {'params': text_params_no_clip, 'lr': opt.learning_rate,},
                {'params': visual_params_no_clip, 'lr': opt.learning_rate,},
                {'params': text_params_clip, 'lr': opt.learning_rate * opt.text_lr_factor,},
                {'params': visual_params_clip,'lr': opt.learning_rate * opt.visual_lr_factor,},
            ], lr=opt.learning_rate, weight_decay=decay_factor)
        else:
            raise ValueError("Unknown precomp_enc_type: {}".format(opt.ext_enc_type))

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        self.Eiters = 0
        self.data_parallel = False

    def set_max_violation(self, max_violation):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), 
                      self.sim_enc.state_dict(), self.criterion.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=True)
        self.txt_enc.load_state_dict(state_dict[1], strict=True)
        self.sim_enc.load_state_dict(state_dict[2], strict=True)
        self.criterion.load_state_dict(state_dict[3], strict=True)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()

    def freeze_backbone(self):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.freeze_backbone()
            else:
                self.img_enc.freeze_backbone()
        elif 'clip' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.freeze_backbone()            
            else:
                self.img_enc.freeze_backbone()
        if 'clip' in self.opt.text_enc_type:
            if isinstance(self.txt_enc, nn.DataParallel):
                self.txt_enc.module.freeze_backbone()
            else:
                self.txt_enc.freeze_backbone()

    def unfreeze_backbone(self, fixed_blocks):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.unfreeze_backbone(fixed_blocks)
            else:
                self.img_enc.unfreeze_backbone(fixed_blocks)
        elif 'clip' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.unfreeze_backbone(fixed_blocks)            
            else:
                self.img_enc.unfreeze_backbone(fixed_blocks)
        if 'clip' in self.opt.text_enc_type:
            if isinstance(self.txt_enc, nn.DataParallel):
                self.txt_enc.module.unfreeze_backbone(fixed_blocks)
            else:
                self.txt_enc.unfreeze_backbone(fixed_blocks)

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc,device_ids=device_ids)
        if self.opt.text_enc_type in ["bert","clip"]:
            self.txt_enc = nn.DataParallel(self.txt_enc,device_ids=device_ids)
            pass
        self.data_parallel = True
        logger.info('Image encoder is data paralleled now.')

    @property
    def is_data_parallel(self):
        return self.data_parallel
    
    # @autocast()
    def forward_emb(self, images, captions, lengths, image_lengths=None):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = images.to(device=img_device)
        captions = captions.to(device=txt_device)
        if isinstance(image_lengths, list):
            image_lengths = torch.Tensor(image_lengths).to(device=img_device)
        else:
            image_lengths = image_lengths.to(device=img_device)
        if isinstance(lengths, list):
            lengths = torch.Tensor(lengths).to(device=txt_device)
        else:
            lengths = lengths.to(device=txt_device)
        with torch.no_grad():
            img_cls, img_embs, image_lengths = self.img_enc(images, image_lengths)
        cap_cls, cap_embs = self.txt_enc(captions, lengths)

        return img_cls, img_embs, cap_cls, cap_embs, image_lengths, lengths

    # @autocast()
    def forward_sim(self, img_cls, img_embs, cap_cls, cap_embs, img_len, cap_len):
        img_cls = img_cls.to(device=sim_device)   
        img_embs =  img_embs.to(device=sim_device)       
        if isinstance(img_len, list):
            img_len = torch.Tensor(img_len)
        img_len = img_len.to(device=sim_device)

        cap_cls = cap_cls.to(device=sim_device) 
        cap_embs =  cap_embs.to(device=sim_device)   
        if isinstance(cap_len, list):
            cap_len = torch.Tensor(cap_len)
        cap_len = cap_len.to(device=sim_device)

        sims = self.sim_enc(img_cls, img_embs, cap_cls, cap_embs, img_len, cap_len)
        return sims

    def forward_loss(self, img_cls, img_embs, cap_cls, cap_embs, img_len, cap_len, matching):
        """Compute the loss given pairs of image and caption embeddings
        """
        sims = self.forward_sim(img_cls, img_embs, cap_cls, cap_embs, img_len, cap_len).to(device=loss_device)
        loss = self.criterion(sims, matching)
        self.logger.update('Le', loss.data.item(), sims.size(0))
        return loss

    def train_emb(self, images, captions, lengths, image_lengths=None, matching=None, warmup_alpha=None):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_cls, img_embs, cap_cls, cap_embs, image_lengths, lengths = self.forward_emb(images, captions, lengths, image_lengths=image_lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_cls, img_embs, cap_cls, cap_embs, image_lengths, lengths, matching)

        if warmup_alpha is not None:
            loss = loss * warmup_alpha

        # compute gradient and update
        loss.backward()

        # torch.autograd.set_detect_anomaly(True)
        # with torch.autograd.detect_anomaly():
        #     loss.backward()

        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()

