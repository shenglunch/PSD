# pip install timm==0.4.12
# pip install einops
# pip install omegaconf
# pip install scikit-image
# 123456a!
# 123456a?
from collections import OrderedDict

import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from .ghostnet import ghostnetv2
from .midas.midas_model import DPTDepthModel
from .depany.depth_anything.dpt import DepthAnything
from .depanyv2.depth_anything_v2.dpt import DepthAnythingV2
# from .depth_pro.src import depth_pro
# from .pvt import PVT
PVT=None
from .submodules import *
from .resnet_cbam import BasicBlock as BBlock
from .resnet_cbam import BasicBlockReZero as BBlockReZero

def get_ghost(pretrained, width, dropout):
    from timm.models import create_model, resume_checkpoint
    model = create_model('ghostnetv2', num_classes=1000, width=width, dropout=dropout, args='TODO')
    # TODO
    # state_dict = torch.load(pretrained)
    # model.load_state_dict(state_dict)       
    return model
 
def get_resnet18(pretrained=None):
    net = torchvision.models.resnet18(pretrained=False)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        net.load_state_dict(state_dict)        
    return net

def get_midas(backbone, pretrained):
    model = DPTDepthModel(backbone=backbone, non_negative=True)
    state_dict = torch.load(pretrained)
    model.load_state_dict(state_dict, strict=False)

    return model

def get_depthanything(backbone, pretrained):
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }

    model = DepthAnything(model_configs[backbone])
    model.load_state_dict(torch.load(pretrained))

    return model

def get_depthanythingv2(backbone, pretrained):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[backbone])
    model.load_state_dict(torch.load(pretrained))

    return model

def get_depthpro(pretrained):
    model, transform = depth_pro.create_model_and_transforms(pretrained)
    return model

def get_pvt(in_chans, patch_size, pretrained):
    model = PVT(in_chans=in_chans, patch_size=patch_size, pretrained=pretrained)
    return model

class SparseDownSample(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, (scale, scale), stride=scale, padding=0, bias=False, padding_mode='zeros')
        if scale == 2:
            self.conv.weight.data = torch.FloatTensor([[[[1, 1],
                                                        [1, 1],]]])
        elif scale == 4:
            self.conv.weight.data = torch.FloatTensor([[[[1, 1, 1, 1],
                                                        [1, 1, 1, 1],
                                                        [1, 1, 1, 1],
                                                        [1, 1, 1, 1],]]])

        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, sparse):
        depth_sum = self.conv(sparse)
        mask_count = self.conv((sparse>0).float())
        depth_avg = depth_sum/(mask_count+1e-5)
        return depth_avg.detach()

class GateFusion(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()

        self.conv1 = nn.Sequential(
            Conv2D(dim1, dim2, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(dim2, dim2, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)

        self.gate = nn.Sequential(
            Conv2D(dim1, dim2, kernel_size=3, stride=1, padding=1, activation='sigmoid', norm='rezero'),)
    
    def forward(self, x):
        x1 = self.conv1(x)
        g = self.gate(x)

        return g*x1

class ResidualBranchRes(nn.Module):
    def __init__(self, cfg, resnet):
        super().__init__()
        self.cfg = cfg

        layer_dim = cfg.RegressBranch.layer_dim
        midas_dim = cfg.RegressBranch.midas_dim
        self.mconv1 = nn.Sequential(
            Conv2D(layer_dim[0], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv2 = nn.Sequential(
            Conv2D(layer_dim[1], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv3 = nn.Sequential(
            Conv2D(layer_dim[2], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv4 = nn.Sequential(
            Conv2D(layer_dim[3], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv = nn.Sequential(
            Conv2D(midas_dim*4, midas_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(midas_dim, midas_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)

        dim = cfg.RegressBranch.dim
        resnet_dim = cfg.RegressBranch.resnet.dim
        self.conv1 = nn.Sequential(
            Conv2D(3, dim, kernel_size=3, stride=2, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(dim, dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(dim, resnet_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)

        self.all_fuse = GateFusion(resnet_dim+midas_dim, resnet_dim)

        self.conv2 = nn.Sequential(
            Conv2D(resnet_dim, resnet_dim, kernel_size=3, stride=2, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(resnet_dim, resnet_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'))
        self.conv3 = nn.Sequential(
            Conv2D(resnet_dim, resnet_dim*2, kernel_size=3, stride=2, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(resnet_dim*2, resnet_dim*2, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'))
        self.conv4 = nn.Sequential(
            BasicBlock(resnet_dim*2, resnet_dim*4, stride=1, activation='lrelu', norm='rezero'),
            BasicBlock(resnet_dim*4, resnet_dim*4, stride=1, activation='lrelu', norm='rezero'))
        self.conv5 = nn.Sequential(
            BasicBlock(resnet_dim*4, resnet_dim*8, stride=1, activation='lrelu', norm='rezero'),
            BasicBlock(resnet_dim*8, resnet_dim*8, stride=1, activation='lrelu', norm='rezero'))

        en_ch = cfg.RegressBranch.resnet.en_ch
        de_ch = cfg.RegressBranch.resnet.de_ch
        self.deconv5 = nn.Sequential(
            Deconv2D(en_ch[4], de_ch[4], kernel_size=3, stride=2, padding=1, output_padding=1, activation='lrelu', norm='rezero'),
            Conv2D(de_ch[4], de_ch[4], kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)
        self.deconv4 = nn.Sequential(
            Deconv2D(de_ch[4]+en_ch[3], de_ch[3], kernel_size=3, stride=2, padding=1, output_padding=1, activation='lrelu', norm='rezero'),
            Conv2D(de_ch[3], de_ch[3], kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)
        self.deconv3 = nn.Sequential(
            Deconv2D(de_ch[3]+en_ch[2], de_ch[2], kernel_size=3, stride=2, padding=1, output_padding=1, activation='lrelu', norm='rezero'),
            Conv2D(de_ch[2], de_ch[2], kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)
        
        last_dim = cfg.RegressBranch.last_dim
        self.regressor = nn.Sequential(
            Conv2D(de_ch[2], last_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(last_dim, 1, kernel_size=3, stride=1, padding=1, activation=None, norm=None),)
        
        self.confhead = nn.Sequential(
            Conv2D(de_ch[2], last_dim//2, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='batch'),
            Conv2D(last_dim//2, 1, kernel_size=3, stride=1, padding=1, activation=None, norm=None),)

    def _make_layer(self, block, planes, *, nums=1, stride=2, padding=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Conv2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0,\
                activation=None, norm='rezero')

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, activation='lrelu', norm='rezero', \
            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, nums):
            layers.append(block(self.inplanes, planes, stride=1, activation='lrelu', norm='rezero', \
                downsample=None))

        return nn.Sequential(*layers)

    def _concat(self, fd, fe, dim=1):
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        if Hd==He and Wd==We:
            return torch.cat((fd, fe), dim=dim)

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=False)

        return torch.cat((fd, fe), dim=dim)

    def forward(self, sparse, depth_scale, residual, mde_feats):           
        x = torch.cat((sparse, depth_scale, residual), dim=1)        
        x1 = self.conv1(x)  # 1/2

        m0 = F.interpolate(self.mconv1(mde_feats[0]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m1 = F.interpolate(self.mconv2(mde_feats[1]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m2 = F.interpolate(self.mconv3(mde_feats[2]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m3 = F.interpolate(self.mconv4(mde_feats[3]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m = self.mconv(torch.cat((m0, m1, m2, m3), dim=1))

        x1 = self.all_fuse(torch.cat((x1, m), dim=1)) # 1/2    32     
        x2 = self.conv2(x1)  # 1/4    32
        x3 = self.conv3(x2)  # 1/8    64
        x4 = self.conv4(x3)  # 1/16   128
        x5 = self.conv5(x4)  # 1/32   256

        d4 = self.deconv5(x5)                   # 1/16  64
        d3 = self.deconv4(self._concat(d4, x4)) # 1/8   32
        d2 = self.deconv3(self._concat(d3, x3)) # 1/4   16

        residual = self.regressor(d2)  
        residual = F.interpolate(residual, size=sparse.shape[2:],\
                        mode="bilinear", align_corners=False)
        confidence = self.confhead(d2)  
        confidence = F.interpolate(confidence, size=sparse.shape[2:],\
                        mode="bilinear", align_corners=False)
        confidence = F.sigmoid(confidence)

        return residual, confidence

class ResidualBranchGh(nn.Module):
    def __init__(self, cfg, ghost):
        super().__init__()
        self.cfg = cfg

        layer_dim = cfg.RegressBranch.layer_dim
        midas_dim = cfg.RegressBranch.midas_dim
        self.mconv1 = nn.Sequential(
            Conv2D(layer_dim[0], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv2 = nn.Sequential(
            Conv2D(layer_dim[1], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv3 = nn.Sequential(
            Conv2D(layer_dim[2], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv4 = nn.Sequential(
            Conv2D(layer_dim[3], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv = nn.Sequential(
            Conv2D(midas_dim*4, midas_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(midas_dim, midas_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)

        dim = cfg.RegressBranch.dim
        ghost_dim = cfg.RegressBranch.ghost.dim
        self.conv1 = nn.Sequential(
            Conv2D(3, dim, kernel_size=3, stride=2, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(dim, dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(dim, ghost_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)
        
        self.all_fuse = GateFusion(ghost_dim+midas_dim, ghost_dim)

        self.ghost_block = ghost.blocks
        self.ghost_layer = cfg.RegressBranch.ghost.layer

        en_ch = cfg.RegressBranch.ghost.en_ch
        de_ch = cfg.RegressBranch.ghost.de_ch
        self.deconv5 = nn.Sequential(
            Deconv2D(en_ch[4], de_ch[4], kernel_size=3, stride=2, padding=1, output_padding=1, activation='lrelu', norm='rezero'),
            Conv2D(de_ch[4], de_ch[4], kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)
        self.deconv4 = nn.Sequential(
            Deconv2D(de_ch[4]+en_ch[3], de_ch[3], kernel_size=3, stride=2, padding=1, output_padding=1, activation='lrelu', norm='rezero'),
            Conv2D(de_ch[3], de_ch[3], kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)
        self.deconv3 = nn.Sequential(
            Deconv2D(de_ch[3]+en_ch[2], de_ch[2], kernel_size=3, stride=2, padding=1, output_padding=1, activation='lrelu', norm='rezero'),
            Conv2D(de_ch[2], de_ch[2], kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)

        last_dim = cfg.RegressBranch.last_dim
        self.regressor = nn.Sequential(
            Conv2D(de_ch[2], last_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(last_dim, 1, kernel_size=3, stride=1, padding=1, activation=None, norm=None),)

    def _make_layer(self, block, planes, *, nums=1, stride=2, padding=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Conv2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0,\
                activation=None, norm='rezero')

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, activation='lrelu', norm='rezero', \
            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, nums):
            layers.append(block(self.inplanes, planes, stride=1, activation='lrelu', norm='rezero', \
                downsample=None))

        return nn.Sequential(*layers)

    def _concat(self, fd, fe, dim=1):
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        if Hd==He and Wd==We:
            return torch.cat((fd, fe), dim=dim)

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=False)
        return torch.cat((fd, fe), dim=dim)

    def forward(self, sparse, depth, residual, mde_feats): 
        x = torch.cat((sparse, depth, residual), dim=1)
        x1 = self.conv1(x)  # 1/2

        m0 = F.interpolate(self.mconv1(mde_feats[0]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m1 = F.interpolate(self.mconv2(mde_feats[1]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m2 = F.interpolate(self.mconv3(mde_feats[2]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m3 = F.interpolate(self.mconv4(mde_feats[3]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m = self.mconv(torch.cat((m0, m1, m2, m3), dim=1))

        x1 = self.all_fuse(torch.cat((x1, m), dim=1))

        xs = []
        for idx, blk in enumerate(self.ghost_block):
            x1 = blk(x1)
            if idx in self.ghost_layer:
                xs.append(x1)
        # 1/2, 1/4, 1/8, 1/16, 1/32
        #  24,  40,  64,  180, 1536 
        x1, x2, x3, x4, x5 = xs
        
        d4 = self.deconv5(x5)                   # 1/16 128
        d3 = self.deconv4(self._concat(d4, x4)) # 1/8  64
        d2 = self.deconv3(self._concat(d3, x3)) # 1/4  32

        residual = self.regressor(d2)  
        residual = F.interpolate(residual, size=sparse.shape[2:],\
                        mode="bilinear", align_corners=False)

        return residual

class ImgBranchRes(nn.Module):
    def __init__(self, cfg, resnet):
        super().__init__()
        self.cfg = cfg

        layer_dim = cfg.CSLBranch.ImgBranch.layer_dim
        midas_dim = cfg.CSLBranch.ImgBranch.midas_dim
        self.mconv1 = nn.Sequential(
            Conv2D(layer_dim[0], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv2 = nn.Sequential(
            Conv2D(layer_dim[1], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv3 = nn.Sequential(
            Conv2D(layer_dim[2], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv4 = nn.Sequential(
            Conv2D(layer_dim[3], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv = nn.Sequential(
            Conv2D(midas_dim*4, midas_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(midas_dim, midas_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)

        dim = cfg.CSLBranch.ImgBranch.dim
        bin_dim = cfg.Net.bin_num
        resnet_dim = cfg.CSLBranch.resnet.dim
        self.conv = nn.Sequential(
            Conv2D(4+bin_dim, dim, kernel_size=3, stride=2, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(dim, dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(dim, resnet_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'))
        
        self.all_fuse = GateFusion(resnet_dim+midas_dim, resnet_dim)

        self.conv1 = nn.Sequential(
            BasicBlock(resnet_dim, resnet_dim, stride=1, activation='lrelu', norm='rezero'),
            BasicBlock(resnet_dim, resnet_dim, stride=1, activation='lrelu', norm='rezero'))
        self.conv2 = nn.Sequential(
            BasicBlock(resnet_dim, resnet_dim*2, stride=2, activation='lrelu', norm='rezero'),
            BasicBlock(resnet_dim*2, resnet_dim*2, stride=1, activation='lrelu', norm='rezero'))
        self.conv3 = nn.Sequential(
            BasicBlock(resnet_dim*2, resnet_dim*4, stride=2, activation='lrelu', norm='rezero'),
            BasicBlock(resnet_dim*4, resnet_dim*4, stride=1, activation='lrelu', norm='rezero'))
        self.conv4 = nn.Sequential(
            BasicBlock(resnet_dim*4, resnet_dim*8, stride=2, activation='lrelu', norm='rezero'),
            BasicBlock(resnet_dim*8, resnet_dim*8, stride=1, activation='lrelu', norm='rezero'))

        en_ch = cfg.CSLBranch.ImgBranch.resnet.en_ch
        de_ch = cfg.CSLBranch.ImgBranch.resnet.de_ch
        self.deconv4 = nn.Sequential(
            Deconv2D(en_ch[3], de_ch[3], kernel_size=3, stride=2, padding=1, output_padding=1, activation='lrelu', norm='rezero'),
            BBlockReZero(de_ch[3], de_ch[3], stride=1, downsample=None, ratio=4),)
        self.deconv3 = nn.Sequential(
            Deconv2D(de_ch[3]+en_ch[2], de_ch[2], kernel_size=3, stride=2, padding=1, output_padding=1, activation='lrelu', norm='rezero'),
            BBlockReZero(de_ch[2], de_ch[2], stride=1, downsample=None, ratio=4),)
        self.deconv2 = nn.Sequential(
            Deconv2D(de_ch[2]+en_ch[1], de_ch[1], kernel_size=3, stride=2, padding=1, output_padding=1, activation='lrelu', norm='rezero'),
            BBlockReZero(de_ch[1], de_ch[1], stride=1, downsample=None, ratio=4),)

        last_dim = cfg.CSLBranch.ImgBranch.last_dim
        self.classifier = nn.Sequential(
            Deconv2D(de_ch[1]+en_ch[0], last_dim, kernel_size=3, stride=2, padding=1, 
                    output_padding=1, activation='lrelu', norm='rezero'),
            Conv2D(last_dim, bin_dim, kernel_size=3, stride=1, padding=1, activation=None, norm=None),)
        
        self.confhead = nn.Sequential(
            Deconv2D(de_ch[1]+en_ch[0], last_dim//3, kernel_size=3, stride=2, padding=1, output_padding=1, activation='lrelu', norm='batch'),
            Conv2D(last_dim//3, 1, kernel_size=3, stride=1, padding=1, activation=None, norm=None),)
        
    def build_simple_position_embedding(self):
        pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.cfg.sparse_net.embed_dim))
        trunc_normal_(pos_embed, std=.02)
        return pos_embed
    
    def get_pos_embed(self, B, H, W):
        if self.training:
            return self.pos_embed
        pos_embed = self.pos_embed
        patch_h, patch_w = self.patch_embed.resolution
        pos_embed = interpolate_pos_encoding(pos_embed, patch_h, patch_w, H, W)
        return pos_embed
    
    def _concat(self, fd, fe, dim=1):
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        if Hd==He and Wd==We:
            return torch.cat((fd, fe), dim=dim)

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=False)
        return torch.cat((fd, fe), dim=dim)

    def forward(self, sparse, depth, residual, label, conf, mde_feats):
        x = torch.cat((sparse, depth, residual, label, conf), dim=1)      
        x1 = self.conv(x)  # 1/2  64

        m0 = F.interpolate(self.mconv1(mde_feats[0]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m1 = F.interpolate(self.mconv2(mde_feats[1]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m2 = F.interpolate(self.mconv3(mde_feats[2]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m3 = F.interpolate(self.mconv4(mde_feats[3]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m = self.mconv(torch.cat((m0, m1, m2, m3), dim=1))

        x1m = self.all_fuse(torch.cat((x1, m), dim=1)) # 1/2    64         

        fe1 = self.conv1(x1)   # 1/2  64
        fe2 = self.conv2(fe1)  # 1/4  128
        fe3 = self.conv3(fe2)  # 1/8  256
        fe4 = self.conv4(fe3)  # 1/16 512

        fd3 = self.deconv4(fe4)                     # 1/8  256
        fd2 = self.deconv3(self._concat(fd3, fe3))  # 1/4  128
        fd1 = self.deconv2(self._concat(fd2, fe2))  # 1/2  64

        fd0_0 = self._concat(fd1, fe1)              # 1/2 128
        
        similarity = self.classifier(fd0_0)  # 1
        confidence = self.confhead(fd0_0)  
        return similarity, confidence, fd0_0

class PixBranchRes(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        dim = cfg.CSLBranch.PixBranch.resnet.dim
        fuse_dim = cfg.CSLBranch.PixBranch.resnet.fuse_dim
 
        self.conv1 = nn.Sequential(
            Conv2D( 4, dim, kernel_size=3, stride=2, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(dim, dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(dim, fuse_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(fuse_dim, fuse_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)
        
        feat_dim = cfg.CSLBranch.PixBranch.resnet.feat_dim
        last_dim = cfg.CSLBranch.PixBranch.resnet.last_dim
        bin_dim = cfg.Net.bin_num

        self.feat2global = nn.Sequential(
            Conv2D(feat_dim, fuse_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),
            Conv2D(fuse_dim, fuse_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)
        self.fuse_global = GateFusion(fuse_dim*2, fuse_dim)
        self.predict_global = nn.Sequential(
            Deconv2D(fuse_dim, last_dim, kernel_size=3, stride=2, padding=1, 
                    output_padding=1, activation='lrelu', norm='rezero'),
            Conv2D(last_dim, 1, kernel_size=3, stride=1, padding=1, activation=None, norm=None),)
        
        self.predict_conf = nn.Sequential(
            Deconv2D(fuse_dim, last_dim//2, kernel_size=3, stride=2, padding=1, 
                    output_padding=1, activation='lrelu', norm='rezero'),
            Conv2D(last_dim//2, 1, kernel_size=3, stride=1, padding=1, activation=None, norm=None),)

    def _concat(self, fd, fe, dim=1):
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        if Hd==He and Wd==We:
            return torch.cat((fd, fe), dim=dim)

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=False)
        return torch.cat((fd, fe), dim=dim)

    def forward(self, sparse, depth, residual, conf, feat):
        x = torch.cat((sparse, depth, residual, conf), dim=1)
        x = self.conv1(x)                  # 1/2  24 

        feat_g = self.feat2global(feat)     # 1/2  24 
        feat_g = self.fuse_global(self._concat(feat_g, x))
        global_offset = self.predict_global(feat_g)
        global_conf = self.predict_conf(feat_g)

        
        global_offset = F.interpolate(global_offset, size=sparse.shape[2:], mode="bilinear", align_corners=False)
        global_conf = F.interpolate(global_conf, size=sparse.shape[2:], mode="bilinear", align_corners=False)
        global_conf = F.sigmoid(global_conf)

        return global_offset, global_conf

class ImgBranchPVT(nn.Module):
    def __init__(self, cfg, pvt):
        super().__init__()
        self.cfg = cfg

        layer_dim = cfg.CSLBranch.ImgBranch.layer_dim
        midas_dim = cfg.CSLBranch.ImgBranch.midas_dim
        self.mconv1 = nn.Sequential(
            Conv2D(layer_dim[0], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv2 = nn.Sequential(
            Conv2D(layer_dim[1], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv3 = nn.Sequential(
            Conv2D(layer_dim[2], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv4 = nn.Sequential(
            Conv2D(layer_dim[3], midas_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),)
        self.mconv = nn.Sequential(
            Conv2D(midas_dim*4, midas_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(midas_dim, midas_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)

        dim = cfg.CSLBranch.ImgBranch.dim
        bin_dim = cfg.Net.bin_num
        pvt_dim = cfg.CSLBranch.pvt.dim

        self.conv = nn.Sequential(
            Conv2D(3+bin_dim, dim, kernel_size=3, stride=2, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(dim, dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(dim, pvt_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'))

        self.all_fuse = GateFusion(pvt_dim+midas_dim, pvt_dim)

        self.former = pvt

        en_ch = cfg.CSLBranch.ImgBranch.pvt.en_ch
        de_ch = cfg.CSLBranch.ImgBranch.pvt.de_ch
        self.deconv6 = nn.Sequential(
            Deconv2D(en_ch[5], de_ch[5], kernel_size=3, stride=2, padding=1, output_padding=1, 
                activation='lrelu', norm='rezero'),
            BBlock(de_ch[5], de_ch[5], stride=1, downsample=None, ratio=4),)
        self.deconv5 = nn.Sequential(
            Deconv2D(de_ch[5]+en_ch[4], de_ch[4], kernel_size=3, stride=2, padding=1, output_padding=1, 
                activation='lrelu', norm='rezero'),
            BBlock(de_ch[4], de_ch[4], stride=1, downsample=None, ratio=4),)
        self.deconv4 = nn.Sequential(
            Deconv2D(de_ch[4]+en_ch[3], de_ch[3], kernel_size=3, stride=2, padding=1, output_padding=1, 
                activation='lrelu', norm='rezero'),
            BBlock(de_ch[3], de_ch[3], stride=1, downsample=None, ratio=4),)
        self.deconv3 = nn.Sequential(
            Deconv2D(de_ch[3]+en_ch[2], de_ch[2], kernel_size=3, stride=2, padding=1, output_padding=1,  
                activation='lrelu', norm='rezero'),
            BBlock(de_ch[2], de_ch[2], stride=1, downsample=None, ratio=4),)
        self.deconv2 = nn.Sequential(
            Deconv2D(de_ch[2]+en_ch[1], de_ch[1], kernel_size=3, stride=2, padding=1, output_padding=1, \
                activation='lrelu', norm='rezero'),
            BBlock(de_ch[1], de_ch[1], stride=1, downsample=None, ratio=4),)

        last_dim = cfg.CSLBranch.ImgBranch.last_dim
        self.classifier = nn.Sequential(
            Deconv2D(de_ch[1]+en_ch[0], last_dim, kernel_size=3, stride=2, padding=1, 
                    output_padding=1, activation='lrelu', norm='rezero'),
            Conv2D(last_dim, bin_dim, kernel_size=3, stride=1, padding=1, activation=None, norm=None),)
        
    def build_simple_position_embedding(self):
        pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.cfg.sparse_net.embed_dim))
        trunc_normal_(pos_embed, std=.02)
        return pos_embed
    
    def get_pos_embed(self, B, H, W):
        if self.training:
            return self.pos_embed
        pos_embed = self.pos_embed
        patch_h, patch_w = self.patch_embed.resolution
        pos_embed = interpolate_pos_encoding(pos_embed, patch_h, patch_w, H, W)
        return pos_embed
    
    def _concat(self, fd, fe, dim=1):
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        if Hd==He and Wd==We:
            return torch.cat((fd, fe), dim=dim)

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=False)
        return torch.cat((fd, fe), dim=dim)

    def forward(self, sparse, depth, residual, label, mde_feats):
        x = torch.cat((sparse, depth, residual, label), dim=1)
        x1 = self.conv(x) # 1/2 

        m0 = F.interpolate(self.mconv1(mde_feats[0]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m1 = F.interpolate(self.mconv2(mde_feats[1]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m2 = F.interpolate(self.mconv3(mde_feats[2]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m3 = F.interpolate(self.mconv4(mde_feats[3]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        m = self.mconv(torch.cat((m0, m1, m2, m3), dim=1))

        x1m = self.all_fuse(torch.cat((x1, m), dim=1)) # 1/2 64

        fe1, fe2, fe3, fe4, fe5, fe6 = self.former(x1) 
        # 1/2  1/4  1/8  1/16  1/32  1/64
        #  64  128   64   128   320   152

        fd5 = self.deconv6(fe6)                     # 1/32  256
        fd4 = self.deconv5(self._concat(fd5, fe5))  # 1/16  160
        fd3 = self.deconv4(self._concat(fd4, fe4))  # 1/8   64
        fd2 = self.deconv3(self._concat(fd3, fe3))  # 1/4   64
        fd1 = self.deconv2(self._concat(fd2, fe2))  # 1/2   64
        fd0_0 = self._concat(fd1, fe1)              # 1/2   128

        similarity = self.classifier(fd0_0)   

        return similarity, fd0_0

class PixBranchPVT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        dim = cfg.CSLBranch.PixBranch.pvt.dim
        fuse_dim = cfg.CSLBranch.PixBranch.pvt.fuse_dim
 
        self.conv1 = nn.Sequential(
            Conv2D( 4, dim, kernel_size=3, stride=2, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(dim, dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(dim, fuse_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),
            Conv2D(fuse_dim, fuse_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)

        feat_dim = cfg.CSLBranch.PixBranch.pvt.feat_dim
        last_dim = cfg.CSLBranch.PixBranch.pvt.last_dim
        bin_dim = cfg.Net.bin_num

        self.feat2local = nn.Sequential(
            Conv2D(feat_dim, fuse_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),
            Conv2D(fuse_dim, fuse_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)
        self.fuse_local = GateFusion(fuse_dim*2, fuse_dim)
        self.predict_local = nn.Sequential(
            Deconv2D(fuse_dim, last_dim, kernel_size=3, stride=2, padding=1, 
                    output_padding=1, activation='lrelu', norm='rezero'),
            Conv2D(last_dim, bin_dim, kernel_size=3, stride=1, padding=1, activation='tanh', norm=None),)
        
        self.feat2global = nn.Sequential(
            Conv2D(feat_dim, fuse_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None),
            Conv2D(fuse_dim, fuse_dim, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='rezero'),)
        self.fuse_global = GateFusion(fuse_dim*2, fuse_dim)
        self.predict_global = nn.Sequential(
            Deconv2D(fuse_dim, last_dim, kernel_size=3, stride=2, padding=1, 
                    output_padding=1, activation='lrelu', norm='rezero'),
            Conv2D(last_dim, 1, kernel_size=3, stride=1, padding=1, activation=None, norm=None),)

    def _concat(self, fd, fe, dim=1):
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        if Hd==He and Wd==We:
            return torch.cat((fd, fe), dim=dim)

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=False)
        return torch.cat((fd, fe), dim=dim)

    def forward(self, sparse, depth, residual, conf, feat):
        x = torch.cat((sparse, depth, residual), dim=1)   
        # x = torch.cat((sparse, depth, conf), dim=1)              
        x = self.conv1(x)                  # 1/2  24 

        feat_l = self.feat2local(feat)     # 1/2  24 
        feat_l = self.fuse_local(self._concat(feat_l, x))
        local_offset = self.predict_local(feat_l)

        feat_g = self.feat2global(feat)     # 1/2  24 
        feat_g = self.fuse_global(self._concat(feat_g, x))
        global_offset = self.predict_global(feat_g)

        return local_offset, global_offset

class MDEBranch(nn.Module):
    def __init__(self, cfg, net):
        super().__init__()
        self.cfg = cfg
        self.net = net

        self.freeze_grad()

    def freeze_grad(self):
        for p in self.net.parameters():
            p.requires_grad = False 

    def forward(self, x, K=None):
        depth_non_scale, path_feats, layer_feats = None, None, None
        if self.cfg.MDEBranch.backbone == 'midas':
            depth_non_scale, path_feats, layer_feats = self.net(x['rgb_mde'])
        elif self.cfg.MDEBranch.backbone == 'depthanything' or self.cfg.MDEBranch.backbone == 'depthanythingv2':
            depth_non_scale, path_feats = self.net(x['rgb_mde'])
            depth_non_scale = depth_non_scale.unsqueeze(1)
        elif self.cfg.MDEBranch.backbone == 'depthpro':
            f = (x['K'][:, 0, 0] + x['K'][:, 1, 1]) / 2.
            depth_non_scale, path_feats = self.net.infer_inv(x['rgb_mde'], f_px=f)
            depth_non_scale = depth_non_scale.float()

        return depth_non_scale, path_feats, layer_feats

class DCNet(nn.Module):
    def __init__(self, cfg):
        super(DCNet, self).__init__()
        self.n = 0
        self.cfg = cfg

        if cfg.MDEBranch.backbone == 'midas':
            midas = get_midas(cfg.MDEBranch.midas.backbone, cfg.MDEBranch.midas.pretrained)
            self.mde = MDEBranch(cfg, midas)
            del midas
        elif cfg.MDEBranch.backbone == 'depthanything':
            depany = get_depthanything(cfg.MDEBranch.depany.backbone, cfg.MDEBranch.depany.pretrained)
            self.mde = MDEBranch(cfg, depany)
            del depany
        elif cfg.MDEBranch.backbone == 'depthanythingv2':
            depanyv2 = get_depthanythingv2(cfg.MDEBranch.depany.backbone, cfg.MDEBranch.depany.pretrained)
            self.mde = MDEBranch(cfg, depanyv2)
            del depanyv2
        elif cfg.MDEBranch.backbone == 'depthpro':
            depthpro = get_depthpro(cfg.MDEBranch.depthpro.pretrained)
            self.mde = MDEBranch(cfg, depthpro)
            del depthpro
        
        if cfg.RegressBranch.backbone == 'ghost':
            ghost = get_ghost(cfg.RegressBranch.ghost.pretrained, cfg.RegressBranch.ghost.width, cfg.RegressBranch.ghost.dropout)
            self.rb = ResidualBranchGh(cfg, ghost)
            del ghost
        elif cfg.RegressBranch.backbone == 'resnet18':
            resnet = get_resnet18(cfg.RegressBranch.resnet.pretrained)
            self.rb = ResidualBranchRes(cfg, resnet)
            del resnet
        
        if cfg.CSLBranch.backbone == 'pvt':
            pvt = get_pvt(cfg.CSLBranch.pvt.dim, cfg.CSLBranch.pvt.patch_size, cfg.CSLBranch.pvt.pretrained)
            self.ib = ImgBranchPVT(cfg, pvt) 
            self.pb = PixBranchPVT(cfg)
            del pvt
        elif cfg.CSLBranch.backbone == 'resnet18':
            resnet = get_resnet18(cfg.CSLBranch.resnet.pretrained)
            self.ib = ImgBranchRes(cfg, resnet) 
            self.pb = PixBranchRes(cfg)
            del resnet

        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.beta1 = nn.Parameter(torch.zeros(1))

        self.alpha2 = nn.Parameter(torch.zeros(1))
        self.beta2 = nn.Parameter(torch.zeros(1))

        self.margin = cfg.Net.margin
        self.num = cfg.Net.bin_num
        
        self.cspn = cfg.Net.cspn
        self.cspn_dwon = cfg.Net.cspn_dwon

        self.NYUv2 = cfg.NYUv2
        self.DIODEi = cfg.DIODEi
        self.SUNRGBD = cfg.SUNRGBD
        self.SCANNET = cfg.SCANNET
        self.MIDDLEBURY = cfg.MIDDLEBURY
        self.HYPERSIM = cfg.HYPERSIM
        self.ETH3D = cfg.ETH3D
        self.VOID1500 = cfg.VOID1500
        self.KITTI = cfg.KITTI
        self.DrivingStereo = cfg.DrivingStereo
        self.DIODEo = cfg.DIODEo
        self.ARGOVERSE = cfg.ARGOVERSE
        self.VKITTI2 = cfg.VKITTI2
        self.DIMLi = cfg.DIMLi
        self.Cityscape = cfg.Cityscape
        self.TOFDC = cfg.TOFDC
        self.HAMMER = cfg.HAMMER
        self.Stanford = cfg.Stanford
        self.KITTI360 = cfg.KITTI360
        self.NK = cfg.NK

        self.scale = cfg.Net.scale
        self.sds = SparseDownSample(self.scale)
        
        self.weight = cfg.train.weight
        self.losses = {}
        self.losses[self.compute_depth_loss] = self.weight.depth_weight
        self.losses[self.compute_confidence_loss] = self.weight.conf_weight
    
    def depth2point(self, depth, K, cam='pinhole'):
        b, c, h, w = depth.shape
        x = torch.linspace(start=0.0, end=w-1, steps=w, device=depth.device).float()
        y = torch.linspace(start=0.0, end=h-1, steps=h, device=depth.device).float()
        # Create H x W grids
        grid_y, grid_x = torch.meshgrid(y, x)
        # Create 3 x H x W grid (x, y, 1)
        grid_xy = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
        grid_xy = torch.unsqueeze(grid_xy, dim=0).repeat(b, 1, 1, 1)
        # Reshape pixel coordinates to N x 3 x (H x W)
        xy_h = grid_xy.reshape(b, 3, -1).float()

        if cam == 'pinhole':
            # K^-1 [x, y, 1] z and reshape back to N x 3 x H x W
            point_homo = torch.matmul(torch.inverse(K.float()), xy_h)
            point_homo = point_homo.reshape(b, 3, h, w) 
            point = point_homo * depth

        elif cam == '360':
            theta = 2 * np.pi * xy_h[:, 0, :] / w
            phi = np.pi * xy_h[:, 1, :] / h
            x = torch.sin(phi) * torch.cos(theta)
            y = torch.sin(phi) * torch.sin(theta)
            z = torch.cos(phi)

            point_homo = torch.cat((x,y,z), dim=1)
            point_homo = point_homo.reshape(b, 3, h, w) 
            point = point_homo * depth
        
        elif cam == 'MEI':
            xy_h = xy_h.reshape(b, 3, h, w) 
            xi = K[:, 0]
            k1 = K[:, 1]
            k2 = K[:, 2]
            p1 = K[:, 3]
            p2 = K[:, 4]
            gamma1 = K[:, 5]
            gamma2 = K[:, 6]
            u0 = K[:, 7]
            v0 = K[:, 8]

            # 计算归一化图像坐标
            x = (xy_h[:, 0, :, :] - u0) / gamma1
            y = (xy_h[:, 1, :, :] - v0) / gamma2

            # 计算径向畸变和切向畸变
            r2 = x**2 + y**2
            r4 = r2**2
            r6 = r2 * r4

            # MEI 模型的畸变校正
            # 计算畸变系数
            distortion = (1 + k1 * r2 + k2 * r4) / (1 + xi * r2) - xi
            x_undistorted = x * distortion + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
            y_undistorted = y * distortion + 2 * p2 * x * y + p1 * (r2 + 2 * y**2)

            # 计算 3D 坐标
            z = torch.sqrt(1 / (1 + xi * r2)) * depth
            x_3d = x_undistorted * z
            y_3d = y_undistorted * z

            point = torch.cat((x_3d, y_3d, z), dim=1)
            point_homo = point / z        
        return point_homo, point

    def point2normal(self, point, size=3):
        b, c, h, w = point.shape

        point_matrix = F.unfold(point, kernel_size=size, stride=1, padding=1, dilation=1)

        # An = b
        matrix_a = point_matrix.reshape(b, 3, size*size, h, w)  # (B, 3, 25, HxW)
        matrix_a = matrix_a.permute(0, 3, 4, 2, 1) # (B, HxW, 25, 3)
        matrix_a_trans = matrix_a.transpose(3, 4)
        matrix_b = torch.ones([1, h, w, size*size, 1]).to(point.device)

        # dot(A.T, A)
        point_multi = torch.matmul(matrix_a_trans, matrix_a)
        matrix_deter = torch.det(point_multi)
        # make inversible
        inverse_condition = torch.ge(matrix_deter, 1e-5)
        inverse_condition = inverse_condition.unsqueeze(-1).unsqueeze(-1)
        inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3)
        # diag matrix to update uninverse
        diag_constant = torch.ones([3], dtype=torch.float32).to(point.device)
        diag_element = torch.diag(diag_constant)
        diag_element = diag_element.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        diag_matrix = diag_element.repeat(1, h, w, 1, 1)
        # inversible matrix
        inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix)
        inv_matrix = torch.inverse(inversible_matrix)

        # compute normal vector use least square
        # n = (A.T A)^-1 A.T b // || (A.T A)^-1 A.T b ||2
        generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)
        normal = F.normalize(generated_norm, p=2, dim=3).squeeze(-1).permute(0, 3, 1, 2)        

        return normal
    
    def depth2distance(self, normal, point):
        distance = (point * normal).sum(1, keepdim=True)
        return distance

    def align_inverse_depth(self, depth_inverse, target, data, mode1='polyfit', mode2='adaptive'):
        def compute_scale_and_shift(prediction, target, mask):
            # system matrix: A = [[a_00, a_01], [a_10, a_11]]
            a_00 = torch.sum(mask * prediction * prediction, (1, 2))
            a_01 = torch.sum(mask * prediction, (1, 2))
            a_11 = torch.sum(mask, (1, 2))

            # right hand side: b = [b_0, b_1]
            b_0 = torch.sum(mask * prediction * target, (1, 2))
            b_1 = torch.sum(mask * target, (1, 2))

            # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
            x_0 = torch.zeros_like(b_0)
            x_1 = torch.zeros_like(b_1)

            det = a_00 * a_11 - a_01 * a_01
            # A needs to be a positive definite matrix.
            valid = det > 0

            x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
            x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

            return x_0, x_1
        
        if mode2=='fixed':
            if data == self.NYUv2.name:
                min_depth = self.NYUv2.min
                max_depth = self.NYUv2.max
            elif data == self.DIODEi.name:
                min_depth = self.DIODEi.min
                max_depth = self.DIODEi.max
            elif data == self.SUNRGBD.name:
                min_depth = self.SUNRGBD.min
                max_depth = self.SUNRGBD.max
            elif data == self.SCANNET.name:
                min_depth = self.SCANNET.min
                max_depth = self.SCANNET.max
            elif data == self.MIDDLEBURY.name:
                min_depth = self.MIDDLEBURY.min
                max_depth = self.MIDDLEBURY.max
            elif data == self.HYPERSIM.name:
                min_depth = self.HYPERSIM.min
                max_depth = self.HYPERSIM.max
            elif data == self.ETH3D.name:
                min_depth = self.ETH3D.min
                max_depth = self.ETH3D.max
            elif data == self.VOID1500.name:
                min_depth = self.VOID1500.min
                max_depth = self.VOID1500.max
            elif data == self.KITTI.name:          
                min_depth = self.KITTI.min
                max_depth = self.KITTI.max
            elif data == self.DrivingStereo.name:
                min_depth = self.DrivingStereo.min
                max_depth = self.DrivingStereo.max
            elif data == self.DIODEo.name:
                min_depth = self.DIODEo.min
                max_depth = self.DIODEo.max
            elif data == self.ARGOVERSE.name:
                min_depth = self.ARGOVERSE.min
                max_depth = self.ARGOVERSE.max
            elif data == self.VKITTI2.name:
                min_depth = self.VKITTI2.min
                max_depth = self.VKITTI2.max
            elif data == self.DIMLi.name:
                min_depth = self.DIMLi.min
                max_depth = self.DIMLi.max
            elif data == self.Cityscape.name:
                min_depth = self.Cityscape.min
                max_depth = self.Cityscape.max
            elif data == self.TOFDC.name:
                min_depth = self.TOFDC.min
                max_depth = self.TOFDC.max
            elif data == self.HAMMER.name:
                min_depth = self.HAMMER.min
                max_depth = self.HAMMER.max
            elif data == self.Stanford.name:
                min_depth = self.Stanford.min
                max_depth = self.Stanford.max
            elif data == self.KITTI360.name:
                min_depth = self.KITTI360.min
                max_depth = self.KITTI360.max
            else:
                raise ValueError
        elif mode2 == 'adaptive':
            mask = target>0
            min_depth = target[mask].min()
            max_depth = target[mask].max()  
        else:
            raise ValueError    

        mask = (target > min_depth) & (target < max_depth)
        target_inverse = torch.zeros_like(target)
        target_inverse[mask == 1] = 1.0 / target[mask == 1]
        
        if mode1=='polyfit':
            scale, shift = compute_scale_and_shift(depth_inverse.squeeze(1), target_inverse.squeeze(1), mask.squeeze(1))
            depth_inverse = scale.reshape(-1, 1, 1, 1) * depth_inverse + shift.reshape(-1, 1, 1, 1)
        elif mode1=='median':
            scale = torch.median(target_inverse[mask]) / (torch.median(depth_inverse[mask]) + 1e-8)
            depth_inverse = depth_inverse * scale
        else:
            raise ValueError

        inverse_cap = 1.0 / max_depth
        depth_inverse[depth_inverse < inverse_cap] = inverse_cap
        
        inverse_cap = 1.0 / min_depth
        depth_inverse[depth_inverse > inverse_cap] = inverse_cap

        depth = 1.0 / depth_inverse
       
        return depth
    
    def align_inverse_depth_nk(self, depth_inverse_nk, target_nk, data_nk, mode1='polyfit', mode2='adaptive'):
        def compute_scale_and_shift(prediction, target, mask):
            # system matrix: A = [[a_00, a_01], [a_10, a_11]]
            a_00 = torch.sum(mask * prediction * prediction, (1, 2))
            a_01 = torch.sum(mask * prediction, (1, 2))
            a_11 = torch.sum(mask, (1, 2))

            # right hand side: b = [b_0, b_1]
            b_0 = torch.sum(mask * prediction * target, (1, 2))
            b_1 = torch.sum(mask * target, (1, 2))

            # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
            x_0 = torch.zeros_like(b_0)
            x_1 = torch.zeros_like(b_1)

            det = a_00 * a_11 - a_01 * a_01
            # A needs to be a positive definite matrix.
            valid = det > 0

            x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
            x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

            return x_0, x_1
        
        depth_nk = []
        for b in range(data_nk.shape[0]):
            b=int(b)
            data = self.KITTI.name if data_nk[b] > 0 else self.NYUv2.name
            
            depth_inverse = depth_inverse_nk[b:(b+1), :, :, :]
            target = target_nk[b:(b+1), :, :, :]

            if mode2=='fixed':
                if data == self.NYUv2.name:
                    min_depth = self.NYUv2.min
                    max_depth = self.NYUv2.max
                elif data == self.KITTI.name:          
                    min_depth = self.KITTI.min
                    max_depth = self.KITTI.max
                else:
                    raise ValueError
            elif mode2 == 'adaptive':
                mask = target>0
                min_depth = target[mask].min()
                max_depth = target[mask].max()
            else:
                raise ValueError

            mask = (target > min_depth) & (target < max_depth)
            target_inverse = torch.zeros_like(target)
            target_inverse[mask == 1] = 1.0 / target[mask == 1]
            
            if mode1=='polyfit':
                scale, shift = compute_scale_and_shift(depth_inverse.squeeze(1), target_inverse.squeeze(1), mask.squeeze(1))
                depth_inverse = scale.reshape(-1, 1, 1, 1) * depth_inverse + shift.reshape(-1, 1, 1, 1)
            elif mode1=='median':
                scale = torch.median(target_inverse[mask]) / (torch.median(depth_inverse[mask]) + 1e-8)
                depth_inverse = depth_inverse * scale
            else:
                raise ValueError

            inverse_cap = 1.0 / max_depth
            depth_inverse[depth_inverse < inverse_cap] = inverse_cap
            
            inverse_cap = 1.0 / min_depth
            depth_inverse[depth_inverse > inverse_cap] = inverse_cap

            depth = 1.0 / depth_inverse

            depth_nk.append(depth)
        
        depth_nk = torch.cat(depth_nk, dim=0)
       
        return depth_nk
 
    def get_depth(self, depth_inverse, target, data):
       
        if data == self.NYUv2.name:
            min_depth = self.NYUv2.min
            max_depth = self.NYUv2.max
        elif data == self.DIODEi.name:
            min_depth = self.DIODEi.min
            max_depth = self.DIODEi.max
        elif data == self.SUNRGBD.name:
            min_depth = self.SUNRGBD.min
            max_depth = self.SUNRGBD.max
        elif data == self.SCANNET.name:
            min_depth = self.SCANNET.min
            max_depth = self.SCANNET.max
        elif data == self.MIDDLEBURY.name:
            min_depth = self.MIDDLEBURY.min
            max_depth = self.MIDDLEBURY.max
        elif data == self.HYPERSIM.name:
            min_depth = self.HYPERSIM.min
            max_depth = self.HYPERSIM.max
        elif data == self.ETH3D.name:
            min_depth = self.ETH3D.min
            max_depth = self.ETH3D.max
        elif data == self.VOID1500.name:
            min_depth = self.VOID1500.min
            max_depth = self.VOID1500.max
        elif data == self.KITTI.name:          
            min_depth = self.KITTI.min
            max_depth = self.KITTI.max
        elif data == self.DrivingStereo.name:
            min_depth = self.DrivingStereo.min
            max_depth = self.DrivingStereo.max
        elif data == self.DIODEo.name:
            min_depth = self.DIODEo.min
            max_depth = self.DIODEo.max
        elif data == self.ARGOVERSE.name:
            min_depth = self.ARGOVERSE.min
            max_depth = self.ARGOVERSE.max
        elif data == self.VKITTI2.name:
            min_depth = self.VKITTI2.min
            max_depth = self.VKITTI2.max
        elif data == self.DIMLi.name:
            min_depth = self.DIMLi.min
            max_depth = self.DIMLi.max
        elif data == self.Cityscape.name:
            min_depth = self.Cityscape.min
            max_depth = self.Cityscape.max
        elif data == self.TOFDC.name:
            min_depth = self.TOFDC.min
            max_depth = self.TOFDC.max
        elif data == self.HAMMER.name:
            min_depth = self.HAMMER.min
            max_depth = self.HAMMER.max
        elif data == self.Stanford.name:
                min_depth = self.Stanford.min
                max_depth = self.Stanford.max
        elif data == self.KITTI360.name:
            min_depth = self.KITTI360.min
            max_depth = self.KITTI360.max
        else:
            raise ValueError

        inverse_cap = 1.0 / max_depth
        depth_inverse[depth_inverse < inverse_cap] = inverse_cap
        
        inverse_cap = 1.0 / min_depth
        depth_inverse[depth_inverse > inverse_cap] = inverse_cap

        depth = 1.0 / depth_inverse
       
        return depth
    
    def align_depth(self, depth, target, data, mode='adaptive'):        
        if mode =='fixed':
            if data == self.NYUv2.name:
                min_depth = self.NYUv2.min
                max_depth = self.NYUv2.max
            elif data == self.DIODEi.name:
                min_depth = self.DIODEi.min
                max_depth = self.DIODEi.max
            elif data == self.SUNRGBD.name:
                min_depth = self.SUNRGBD.min
                max_depth = self.SUNRGBD.max
            elif data == self.SCANNET.name:
                min_depth = self.SCANNET.min
                max_depth = self.SCANNET.max
            elif data == self.MIDDLEBURY.name:
                min_depth = self.MIDDLEBURY.min
                max_depth = self.MIDDLEBURY.max
            elif data == self.HYPERSIM.name:
                min_depth = self.HYPERSIM.min
                max_depth = self.HYPERSIM.max
            elif data == self.ETH3D.name:
                min_depth = self.ETH3D.min
                max_depth = self.ETH3D.max
            elif data == self.VOID1500.name:
                min_depth = self.VOID1500.min
                max_depth = self.VOID1500.max
            elif data == self.KITTI.name:          
                min_depth = self.KITTI.min
                max_depth = self.KITTI.max
            elif data == self.DrivingStereo.name:
                min_depth = self.DrivingStereo.min
                max_depth = self.DrivingStereo.max
            elif data == self.DIODEo.name:
                min_depth = self.DIODEo.min
                max_depth = self.DIODEo.max
            elif data == self.ARGOVERSE.name:
                min_depth = self.ARGOVERSE.min
                max_depth = self.ARGOVERSE.max
            elif data == self.VKITTI2.name:
                min_depth = self.VKITTI2.min
                max_depth = self.VKITTI2.max
            elif data == self.DIMLi.name:
                min_depth = self.DIMLi.min
                max_depth = self.DIMLi.max
            elif data == self.Cityscape.name:
                min_depth = self.Cityscape.min
                max_depth = self.Cityscape.max
            elif data == self.TOFDC.name:
                min_depth = self.TOFDC.min
                max_depth = self.TOFDC.max
            elif data == self.HAMMER.name:
                min_depth = self.HAMMER.min
                max_depth = self.HAMMER.max
            elif data == self.Stanford.name:
                min_depth = self.Stanford.min
                max_depth = self.Stanford.max
            elif data == self.KITTI360.name:
                min_depth = self.KITTI360.min
                max_depth = self.KITTI360.max
            else:
                raise ValueError
        elif mode == 'adaptive':
            mask = target>0
            min_depth = target[mask].min()
            max_depth = target[mask].max()  
        else:
            raise ValueError    

        mask = (target > min_depth) & (target < max_depth)
        
        scale = torch.median(target[mask]) / (torch.median(depth[mask]) + 1e-8)
        depth = depth * scale

        depth[depth < min_depth] = min_depth
        depth[depth > max_depth] = max_depth
       
        return depth

    def align_depth_nk(self, depth_nk, target_nk, data_nk, mode='adaptive'):
        depth_nk2 = []
        for b in range(data_nk.shape[0]):
            b=int(b)
            data = self.KITTI.name if data_nk[b] > 0 else self.NYUv2.name
            
            depth = depth_nk[b:(b+1), :, :, :]
            target = target_nk[b:(b+1), :, :, :]

            if mode =='fixed':
                if data == self.NYUv2.name:
                    min_depth = self.NYUv2.min
                    max_depth = self.NYUv2.max
                elif data == self.KITTI.name:          
                    min_depth = self.KITTI.min
                    max_depth = self.KITTI.max
                else:
                    raise ValueError
            elif mode == 'adaptive':
                mask = target>0
                min_depth = target[mask].min()
                max_depth = target[mask].max()  
            else:
                raise ValueError    

            mask = (target > min_depth) & (target < max_depth)
            
            scale = torch.median(target[mask]) / (torch.median(depth[mask]) + 1e-8)
            depth = depth * scale

            depth[depth < min_depth] = min_depth
            depth[depth > max_depth] = max_depth
            depth_nk2.append(depth)

        depth_nk2 = torch.cat(depth_nk2, dim=0)
        return depth_nk2

    def compute_affinity(self, feat, k, distance='l2'):
        b, c, h, w = feat.shape
        feat = F.unfold(feat, kernel_size=k, padding=1, stride=1) # b, c*k*k, l
        feat = feat.reshape(b, c, k*k, -1)     # b, c, k*k, l
        feat_border = torch.zeros((b, c, k*k-1, h*w)).to(feat.device)
        feat_border[:, :, :(k+1), :] = feat[:, :, :(k+1), :]
        feat_border[:, :, (k+1):, :] = feat[:, :, (k+2):, :]
        feat_center = feat[:, :, (k+1):(k+2), :]

        if distance=='l2':
            affinity = feat_border - feat_center  # b, c, k*k-1, l
            affinity = torch.norm(affinity, p=2, dim=1).reshape(b, -1, h, w) # b, k*k-1, l
        elif distance=='cosine':
            feat_border = F.normalize(feat_border, p=2, dim=1)
            feat_center = F.normalize(feat_center, p=2, dim=1)
            affinity = feat_border * feat_center  # b, c, k*k-1, l
            affinity = affinity.sum(dim=1).reshape(b, -1, h, w) # b, k*k-1, l

        return affinity

    def affinity_normalization(self, sum_conv, affinity):
        affinity = affinity.to(sum_conv.weight.dtype)

        gate1_wb_cmb = affinity.narrow(1, 0    , 1)
        gate2_wb_cmb = affinity.narrow(1, 1 * 1, 1)
        gate3_wb_cmb = affinity.narrow(1, 2 * 1, 1)
        gate4_wb_cmb = affinity.narrow(1, 3 * 1, 1)
        gate5_wb_cmb = affinity.narrow(1, 4 * 1, 1)
        gate6_wb_cmb = affinity.narrow(1, 5 * 1, 1)
        gate7_wb_cmb = affinity.narrow(1, 6 * 1, 1)
        gate8_wb_cmb = affinity.narrow(1, 7 * 1, 1)

        # gate1:left_top, gate2:center_top, gate3:right_top
        # gate4:left_center,              , gate5: right_center
        # gate6:left_bottom, gate7: center_bottom, gate8: right_bottm

        # top pad
        left_top_pad = nn.ZeroPad2d((0,2,0,2))
        gate1_wb_cmb = left_top_pad(gate1_wb_cmb).unsqueeze(1)

        center_top_pad = nn.ZeroPad2d((1,1,0,2))
        gate2_wb_cmb = center_top_pad(gate2_wb_cmb).unsqueeze(1)

        right_top_pad = nn.ZeroPad2d((2,0,0,2))
        gate3_wb_cmb = right_top_pad(gate3_wb_cmb).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0,2,1,1))
        gate4_wb_cmb = left_center_pad(gate4_wb_cmb).unsqueeze(1)

        right_center_pad = nn.ZeroPad2d((2,0,1,1))
        gate5_wb_cmb = right_center_pad(gate5_wb_cmb).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0,2,2,0))
        gate6_wb_cmb = left_bottom_pad(gate6_wb_cmb).unsqueeze(1)

        center_bottom_pad = nn.ZeroPad2d((1,1,2,0))
        gate7_wb_cmb = center_bottom_pad(gate7_wb_cmb).unsqueeze(1)

        right_bottm_pad = nn.ZeroPad2d((2,0,2,0))
        gate8_wb_cmb = right_bottm_pad(gate8_wb_cmb).unsqueeze(1)

        gate_wb = torch.cat((gate1_wb_cmb,gate2_wb_cmb,gate3_wb_cmb,gate4_wb_cmb,
                             gate5_wb_cmb,gate6_wb_cmb,gate7_wb_cmb,gate8_wb_cmb), 1)

        # normalize affinity using their abs sum
        gate_wb_abs = torch.abs(gate_wb)
        abs_weight = sum_conv(gate_wb_abs)

        gate_wb = torch.div(gate_wb, abs_weight)
        gate_sum = sum_conv(gate_wb)

        gate_sum = gate_sum.squeeze(1)
        gate_sum = gate_sum[:, :, 1:-1, 1:-1]

        return gate_wb, 1.0 - gate_sum
    
    def affinity_zeroization(self, sum_conv, affinity):
        gate1_wb_cmb = affinity.narrow(1, 0    , 1)
        gate2_wb_cmb = affinity.narrow(1, 1 * 1, 1)
        gate3_wb_cmb = affinity.narrow(1, 2 * 1, 1)
        gate4_wb_cmb = affinity.narrow(1, 3 * 1, 1)
        gate5_wb_cmb = affinity.narrow(1, 4 * 1, 1)
        gate6_wb_cmb = affinity.narrow(1, 5 * 1, 1)
        gate7_wb_cmb = affinity.narrow(1, 6 * 1, 1)
        gate8_wb_cmb = affinity.narrow(1, 7 * 1, 1)

        # gate1:left_top, gate2:center_top, gate3:right_top
        # gate4:left_center,              , gate5: right_center
        # gate6:left_bottom, gate7: center_bottom, gate8: right_bottm

        # top pad
        left_top_pad = nn.ZeroPad2d((0,2,0,2))
        gate1_wb_cmb = left_top_pad(gate1_wb_cmb).unsqueeze(1)

        center_top_pad = nn.ZeroPad2d((1,1,0,2))
        gate2_wb_cmb = center_top_pad(gate2_wb_cmb).unsqueeze(1)

        right_top_pad = nn.ZeroPad2d((2,0,0,2))
        gate3_wb_cmb = right_top_pad(gate3_wb_cmb).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0,2,1,1))
        gate4_wb_cmb = left_center_pad(gate4_wb_cmb).unsqueeze(1)

        right_center_pad = nn.ZeroPad2d((2,0,1,1))
        gate5_wb_cmb = right_center_pad(gate5_wb_cmb).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0,2,2,0))
        gate6_wb_cmb = left_bottom_pad(gate6_wb_cmb).unsqueeze(1)

        center_bottom_pad = nn.ZeroPad2d((1,1,2,0))
        gate7_wb_cmb = center_bottom_pad(gate7_wb_cmb).unsqueeze(1)

        right_bottm_pad = nn.ZeroPad2d((2,0,2,0))
        gate8_wb_cmb = right_bottm_pad(gate8_wb_cmb).unsqueeze(1)

        gate_wb = torch.cat((gate1_wb_cmb,gate2_wb_cmb,gate3_wb_cmb,gate4_wb_cmb,
                             gate5_wb_cmb,gate6_wb_cmb,gate7_wb_cmb,gate8_wb_cmb), 1)

        # zeroization affinity using their abs sum
        gate_wb_abs = torch.abs(gate_wb)
        abs_weight = sum_conv(gate_wb_abs)

        gate_wb = torch.div(gate_wb, abs_weight)
        gate_sum = sum_conv(gate_wb)

        gate_sum = gate_sum.squeeze(1)
        gate_sum = gate_sum[:, :, 1:-1, 1:-1]

        return gate_wb, gate_sum

    def pad_depth(self, depth):
        # top pad
        left_top_pad = nn.ZeroPad2d((0,2,0,2))
        depth_1 = left_top_pad(depth).unsqueeze(1)
        center_top_pad = nn.ZeroPad2d((1,1,0,2))
        depth_2 = center_top_pad(depth).unsqueeze(1)
        right_top_pad = nn.ZeroPad2d((2,0,0,2))
        depth_3 = right_top_pad(depth).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0,2,1,1))
        depth_4 = left_center_pad(depth).unsqueeze(1)
        right_center_pad = nn.ZeroPad2d((2,0,1,1))
        depth_5 = right_center_pad(depth).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0,2,2,0))
        depth_6 = left_bottom_pad(depth).unsqueeze(1)
        center_bottom_pad = nn.ZeroPad2d((1,1,2,0))
        depth_7 = center_bottom_pad(depth).unsqueeze(1)
        right_bottm_pad = nn.ZeroPad2d((2,0,2,0))
        depth_8 = right_bottm_pad(depth).unsqueeze(1)

        result_depth = torch.cat((depth_1, depth_2, depth_3, depth_4,
                                  depth_5, depth_6, depth_7, depth_8), 1)
        return result_depth
    
    def CSPN(self, sparse, dense, feat, iteration=None):
        k = self.cspn.kernel
        if iteration is not None:
            iter = iteration
        else:
            iter = self.cspn.iteration

        B, C, H, W = dense.shape
        mask = sparse > 0
        depth = (1 - mask.float()) * dense + mask.float() * sparse

        
        affinity = self.compute_affinity(feat, k, distance='cosine')
        sum_conv = nn.Conv3d(in_channels=k*k-1, out_channels=1,
                             kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)
        weight = torch.ones(1, k*k-1, 1, 1, 1).to(feat.device)
        sum_conv.weight = nn.Parameter(weight)
        for param in sum_conv.parameters():
            param.requires_grad = False

        neighbor_w, center_w = self.affinity_normalization(sum_conv, affinity) # (b, 8, 1, h ,w) (b, 1, h ,w) 
        raw_depth_input = depth
        result_depth = raw_depth_input

        for i in range(iter):
            result_depth = (1 - mask.float()) * result_depth + mask.float() * sparse
            result_depth = self.pad_depth(result_depth)       
            tmp = neighbor_w * result_depth
            tmp = tmp.to(sum_conv.weight.dtype)
            neigbor_weighted_sum = sum_conv(tmp)
            neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)
            neigbor_weighted_sum = neigbor_weighted_sum[:, :, 1:-1, 1:-1]
            result_depth = neigbor_weighted_sum
            result_depth = center_w * raw_depth_input + result_depth

        depth = result_depth
        return depth
    
    def CSPN_down(self, sparse, dense, feat):
        k = self.cspn_dwon.kernel
        iter = self.cspn_dwon.iteration

        B, C, H, W = dense.shape
        mask = sparse > 0
        depth = (1 - mask.float()) * dense + mask.float() * sparse

        
        affinity = self.compute_affinity(feat, k, distance='cosine')
        sum_conv = nn.Conv3d(in_channels=k*k-1, out_channels=1,
                             kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)
        weight = torch.ones(1, k*k-1, 1, 1, 1).to(feat.device)
        sum_conv.weight = nn.Parameter(weight)
        for param in sum_conv.parameters():
            param.requires_grad = False

        neighbor_w, center_w = self.affinity_normalization(sum_conv, affinity) # (b, 8, 1, h ,w) (b, 1, h ,w) 
        raw_depth_input = depth
        result_depth = raw_depth_input

        for i in range(iter):
            result_depth = (1 - mask.float()) * result_depth + mask.float() * sparse
            result_depth = self.pad_depth(result_depth)       
            tmp = neighbor_w * result_depth
            tmp = tmp.to(sum_conv.weight.dtype)
            neigbor_weighted_sum = sum_conv(tmp)
            neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)
            neigbor_weighted_sum = neigbor_weighted_sum[:, :, 1:-1, 1:-1]
            result_depth = neigbor_weighted_sum
            result_depth = center_w * raw_depth_input + result_depth

        depth = result_depth
        return depth
    
    def knnv1(self, feat, point, depth, sparse, data):
        b, c, h, w = feat.shape

        # 近邻数量, 更新点比率, 近邻点总数, 选近邻点方法
        knn, knn_rate, knn_candidate, knn_mode = self.get_knn(data)
        if knn_candidate == 0:
            mask = sparse > 1e-4
            knn_candidate = mask.sum(dim=(1,2,3)).min()
            # knn_candidate = 9999999
            # for i in range(b):
            #     num = int((sparse[i] > 0).sum())
            #     knn_candidate = knn_candidate if num > knn_candidate else num

        # 近邻点 id号
        idx_nz = []
        for i in range(b):
            idx = self.get_knn_candidate(sparse[i], knn_candidate)
            idx_nz.append(idx.permute(1, 0))
        idx_nz = torch.cat(idx_nz, dim=0)
        # sparse # b, m, 3; b, m, c; b, m, 1; b, n, 3
        # 近邻点 点云，特征，稀疏深度值，法线
        point_nz, feat_nz, sparse_nz, normal_nz \
            = self.get_idx(idx_nz, feat, point, sparse)

        # select # b, n, 3; b, n, c; b, n, 1; b, n 3
        # 更新点 id号
        if knn_mode == 'random':
            idx_sl = self.select_random_pointv1(knn_rate, sparse, b, h*w)
        elif knn_mode == 'fps':
            idx_sl = self.select_fps_pointv1(knn_rate, point, sparse)
        # 更新点 点云，特征，稀疏深度值，法线
        point_sl, feat_sl, depth_sl, normal_sl \
            = self.get_idx(idx_sl, feat, point, depth)

        n = point_sl.shape[1] # number of select
        m = point_nz.shape[1] # number of sparse (non-zero)

        # first point euclidean distance
        # 更新点 根据点云 选择近邻点
        eu_dist = point_sl.unsqueeze(1).repeat(1, m, 1, 1) - point_nz.unsqueeze(2) # b, m, n, 3
        eu_dist = torch.norm(eu_dist, p=2, dim=-1) # b, m, n
        eu_affinity, eu_index = torch.topk(eu_dist, k=knn*2, dim=1, largest=False) # b, k*2, n   return smallest
        sparse_nz = sparse_nz.unsqueeze(2).repeat(1, 1, n, 1) # b, m, n, 1
        eu_sparse = sparse_nz.gather(1, eu_index.unsqueeze(-1).repeat(1, 1, 1, 1)) # b, 2*k, n, 1

        # second consine similarity
        # 更新点 根据特征 选择近邻点
        feat_nz = feat_nz.unsqueeze(2).repeat(1, 1, n, 1) 
        eu_feat = feat_nz.gather(1, eu_index.unsqueeze(-1).repeat(1, 1, 1, c))
        eu_feat = eu_feat.reshape(b, 2*knn, n, c) # b, 2*k, n, c
        
        feat_sl = F.normalize(feat_sl, p=2, dim=-1)
        eu_feat = F.normalize(eu_feat, p=2, dim=-1)

        affinity = torch.sum(eu_feat * feat_sl.unsqueeze(1), dim=-1)    # b, 2*k, n
        k_affinity, k_index = torch.topk(affinity, k=knn, dim=1, largest=False) # b, k, n
        k_sparse = eu_sparse.gather(1, k_index.unsqueeze(-1).repeat(1, 1, 1, 1)) # b, k, n, 1

        # 更新点 深度值更新
        # momentum propagate
        momentum = 1.0
        knn_sim_sum = torch.sum(k_affinity, dim=1, keepdim=True)     # b, k, n
        knn_sim_div = torch.div(k_affinity, knn_sim_sum)             # b, k, n

        new = torch.sum(k_sparse * knn_sim_div.unsqueeze(-1), dim=1) # b, n, 1
        new = depth_sl * (1.0 - momentum) + new * momentum
        depth = depth.float()
        new = new.float()
        # update
        depth = depth.reshape(b, 1, h*w).permute(0, 2, 1)            # b, hw, 1
        depth = depth.scatter(1, idx_sl.unsqueeze(-1), new).reshape(b, h, w, 1).permute(0, 3, 1, 2) # b, 1, h, w

        return depth

    def get_knn(self, data):
        if data == self.NYUv2.name:
            knn_rate = self.NYUv2.knn_rate
            knn = self.NYUv2.knn
            knn_candidate = self.NYUv2.knn_candidate
        elif data == self.DIODEi.name:
            knn_rate = self.DIODEi.knn_rate
            knn = self.DIODEi.knn
            knn_candidate = self.DIODEi.knn_candidate
        elif data == self.SUNRGBD.name:
            knn_rate = self.SUNRGBD.knn_rate
            knn = self.SUNRGBD.knn
            knn_candidate = self.SUNRGBD.knn_candidate
        elif data == self.SCANNET.name:
            knn_rate = self.SCANNET.knn_rate
            knn = self.SCANNET.knn
            knn_candidate = self.SCANNET.knn_candidate
        elif data == self.MIDDLEBURY.name:
            knn_rate = self.MIDDLEBURY.knn_rate
            knn = self.MIDDLEBURY.knn
            knn_candidate = self.MIDDLEBURY.knn_candidate
        elif data == self.HYPERSIM.name:
            knn_rate = self.HYPERSIM.knn_rate
            knn = self.HYPERSIM.knn
            knn_candidate = self.HYPERSIM.knn_candidate
        elif data == self.ETH3D.name:
            knn_rate = self.ETH3D.knn_rate
            knn = self.ETH3D.knn
            knn_candidate = self.ETH3D.knn_candidate
        elif data == self.VOID1500.name:
            knn_rate = self.VOID1500.knn_rate
            knn = self.VOID1500.knn
            knn_candidate = self.VOID1500.knn_candidate
        elif data == self.KITTI.name:
            knn_rate = self.KITTI.knn_rate
            knn = self.KITTI.knn
            knn_candidate = self.KITTI.knn_candidate
            if self.training:
                knn_rate = self.KITTI.knn_rate_train
                knn_candidate = self.KITTI.knn_candidate_train
        elif data == self.DrivingStereo.name:
            knn_rate = self.DrivingStereo.knn_rate
            knn = self.DrivingStereo.knn
            knn_candidate = self.DrivingStereo.knn_candidate
        elif data == self.DIODEo.name:
            knn_rate = self.DIODEo.knn_rate
            knn = self.DIODEo.knn
            knn_candidate = self.DIODEo.knn_candidate
        elif data == self.ARGOVERSE.name:
            knn_rate = self.ARGOVERSE.knn_rate
            knn = self.ARGOVERSE.knn
            knn_candidate = self.ARGOVERSE.knn_candidate
        elif data == self.VKITTI2.name:
            knn_rate = self.VKITTI2.knn_rate
            knn = self.VKITTI2.knn
            knn_candidate = self.VKITTI2.knn_candidate
        elif data == self.NK.name:
            knn_rate = self.NK.knn_rate
            knn = self.NK.knn
            knn_candidate = self.NK.knn_candidate
        else:
            knn_rate = 0.5
            knn = 3
            knn_candidate = 0
        
        mode = 'random'
        return knn, knn_rate, knn_candidate, mode

    def select_random_pointv1(self, rate, sparse, batch, size):
        mask = sparse > 1e-6
        sparse_num = mask.sum(dim=(1,2,3)).max()
        sample_num = int((size - sparse_num) *rate)

        idx_sl = []
        for i in range(batch):
            idx = torch.nonzero(sparse[i].view(-1) <= 0, as_tuple=False)
            idx_sample = torch.randperm(len(idx))[:sample_num]
            idx = idx[idx_sample[:]]
            idx_sl.append(idx.permute(1, 0))
        idx_sl = torch.cat(idx_sl, dim=0)
        return idx_sl 
  
    def select_fps_pointv1(self, rate, point, sparse):
        b, c, h, w = point.shape

        idx = self.farthest_point_sample(point, sparse, int(h*w*rate))

        return idx
    
    def farthest_point_sample(self, point, sparse, num_points):
        """
        Input:
            point: pointcloud data, [B, 3, H, W]
            sparse: valid value, [B, 1, H, W]
            num_points: number of samples
        Return:
            centroids: sampled pointcloud index, [B, num_points]
        """
        device = point.device
        B, C, H, W = point.shape
        N = H * W
        xyz = point.view(B, -1, 3)
        spz = sparse.view(B, -1)

        distance = torch.ones(B, N).to(device) * 1e10
        weight = torch.where(spz>0, torch.zeros(B, N).to(device), torch.ones(B, N).to(device))
        centroids = torch.zeros(B, num_points, dtype=torch.long).to(device)
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)

        for i in range(num_points):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1) * weight
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        return centroids

    def get_knn_candidate(self, sparse, num_sample):
        channel, height, width = sparse.shape

        assert channel == 1

        idx_nnz = torch.nonzero(sparse.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        return idx_nnz

    def get_idx(self, idx, feat, point, sparse, normal=None):
        b, c, h, w = feat.shape

        point = point.reshape(b, 3, h*w).permute(0, 2, 1)              # b, hw, 3
        point_nz = point.gather(1, idx.unsqueeze(2).repeat(1, 1, 3))   # b, len_idx, 3

        feat = feat.reshape(b, c, h*w).permute(0, 2, 1)                # b, hw, c
        feat_nz = feat.gather(1, idx.unsqueeze(2).repeat(1, 1, c))     # b, len_idx, c

        sparse = sparse.reshape(b, 1, h*w).permute(0, 2, 1)            # b, hw, 1 
        sparse_nz = sparse.gather(1, idx.unsqueeze(2).repeat(1, 1, 1)) # b, len_idx, 1
        
        if normal is not None:
            normal = normal.reshape(b, 3, h*w).permute(0, 2, 1)            # b, hw, 1 
            normal_nz = normal.gather(1, idx.unsqueeze(2).repeat(1, 1, 3)) # b, len_idx, 1
        else:
            normal_nz = None

        return point_nz, feat_nz, sparse_nz, normal_nz

    def post_processing(self, feat, dense, sparse, data):
        if data == self.NYUv2.name:
            iteration = self.NYUv2.iteration
        elif data == self.DIODEi.name:
            iteration = self.DIODEi.iteration
        elif data == self.SUNRGBD.name:
            iteration = self.SUNRGBD.iteration
        elif data == self.SCANNET.name:
            iteration = self.SCANNET.iteration
        elif data == self.MIDDLEBURY.name:
            iteration = self.MIDDLEBURY.iteration
        elif data == self.HYPERSIM.name:
            iteration = self.HYPERSIM.iteration
        elif data == self.ETH3D.name:
            iteration = self.ETH3D.iteration
        elif data == self.VOID1500.name:
            iteration = self.VOID1500.iteration
        elif data == self.KITTI.name:  
            iteration = self.KITTI.iteration
        elif data == self.DrivingStereo.name:
            iteration = self.DrivingStereo.iteration
        elif data == self.DIODEo.name:
            iteration = self.DIODEo.iteration
        elif data == self.ARGOVERSE.name:
            iteration = self.ARGOVERSE.iteration
        elif data == self.VKITTI2.name:
            iteration = self.VKITTI2.iteration
        elif data == self.DIMLi.name:
            iteration = self.DIMLi.iteration
        elif data == self.Cityscape.name:
            iteration = self.Cityscape.iteration
        elif data == self.TOFDC.name:
            iteration = self.TOFDC.iteration
        elif data == self.HAMMER.name:
            iteration = self.HAMMER.iteration
        else:
            iteration = -1

        if iteration < 0:
            return dense
        elif iteration == 0:
            mask = sparse > 0
            dense = dense * (1-mask.float()) + sparse
            return dense
        else:
            feat = F.interpolate(feat.unsqueeze(1), size=(32, dense.shape[2], dense.shape[3]), mode="trilinear", align_corners=False).squeeze(1)
            depth_cspn = self.CSPN(sparse, dense, feat, iteration=iteration)
            return depth_cspn.detach()

    def dual_diffusion(self, feat, ip_median, depth_polyfit, sparse, inputs, data, cam='pinhole'):
        cam = 'pinhole'
        K = inputs['K']
        if data == self.NYUv2.name:
            diffusion = self.NYUv2.diffusion
        elif data == self.DIODEi.name:
            diffusion = self.DIODEi.diffusion
        elif data == self.SUNRGBD.name:
            diffusion = self.SUNRGBD.diffusion
        elif data == self.SCANNET.name:
            diffusion = self.SCANNET.diffusion
        elif data == self.MIDDLEBURY.name:
            diffusion = self.MIDDLEBURY.diffusion
        elif data == self.HYPERSIM.name:
            diffusion = self.HYPERSIM.diffusion
        elif data == self.ETH3D.name:
            diffusion = self.ETH3D.diffusion
        elif data == self.VOID1500.name:
            diffusion = self.VOID1500.diffusion
        elif data == self.KITTI.name:  
            diffusion = self.KITTI.diffusion
        elif data == self.DrivingStereo.name:
            diffusion = self.DrivingStereo.diffusion
        elif data == self.DIODEo.name:
            diffusion = self.DIODEo.diffusion
        elif data == self.ARGOVERSE.name:
            diffusion = self.ARGOVERSE.diffusion
        elif data == self.VKITTI2.name:
            diffusion = self.VKITTI2.diffusion
        elif data == self.DIMLi.name:
            diffusion = self.DIMLi.diffusion
        elif data == self.Cityscape.name:
            diffusion = self.Cityscape.diffusion
        elif data == self.TOFDC.name:
            diffusion = self.TOFDC.diffusion
        elif data == self.HAMMER.name:
            diffusion = self.HAMMER.diffusion
        elif data == self.Stanford.name:
            cam = '360'
            diffusion = self.Stanford.diffusion
        elif data == self.KITTI360.name:
            cam = 'MEI'
            K = inputs['M']
            diffusion = self.KITTI360.diffusion
        else:
            diffusion = '3D-2D'
            
        if diffusion == '3D-2D':
            # knn & spn
            B, C, H, W = sparse.shape
            feat = feat.unsqueeze(1)
            point_homo, point = self.depth2point(depth_polyfit, K, cam=cam)
            ip_median_down = F.interpolate(ip_median, size=(H//self.scale, W//self.scale), mode="bicubic", align_corners=False)
            point_down = F.interpolate(point, size=(H//self.scale, W//self.scale), mode="bicubic", align_corners=False)
            sparse_down = self.sds(sparse)

            cost_feat = F.interpolate(feat, size=(9, H//self.scale, W//self.scale), mode="trilinear", align_corners=False).squeeze(1)
            cost_feat = torch.cat((point_down, cost_feat), dim=1)
            depth_knn_m_dwon = self.knnv1(cost_feat, point_down, ip_median_down.clone(), sparse_down, data).detach()
            depth_cspn_m_dwon = self.CSPN_down(sparse_down, depth_knn_m_dwon, cost_feat).detach()
            depth_cspn_m_dwon = F.interpolate(depth_cspn_m_dwon, size=(H, W), mode="bicubic", align_corners=False)
            cost_feat = F.interpolate(feat, size=(16, H, W), mode="trilinear", align_corners=False).squeeze(1)
            depth_cspn_m = self.CSPN(sparse, depth_cspn_m_dwon, cost_feat).detach()
        elif diffusion == '3D+2D':
            B, C, H, W = sparse.shape
            feat = feat.unsqueeze(1)
            # only 3D
            point_homo, point = self.depth2point(depth_polyfit, K, cam=cam)
            ip_median_down = F.interpolate(ip_median, size=(H//self.scale, W//self.scale), mode="bicubic", align_corners=False)
            point_down = F.interpolate(point, size=(H//self.scale, W//self.scale), mode="bicubic", align_corners=False)
            sparse_down = self.sds(sparse)
            cost_feat = F.interpolate(feat, size=(9, H//self.scale, W//self.scale), mode="trilinear", align_corners=False).squeeze(1)
            cost_feat = torch.cat((point_down, cost_feat), dim=1)
            depth_knn_m_dwon = self.knnv1(cost_feat, point_down, ip_median_down.clone(), sparse_down, data).detach()
            
            # only 2D
            depth_cspn_m_dwon = self.CSPN_down(sparse_down, ip_median_down, cost_feat).detach()
            depth_cspn_m_dwon = F.interpolate(depth_cspn_m_dwon, size=(H, W), mode="bicubic", align_corners=False)
            cost_feat = F.interpolate(feat, size=(16, H, W), mode="trilinear", align_corners=False).squeeze(1)
            depth_cspn_m = self.CSPN(sparse, depth_cspn_m_dwon, cost_feat).detach()
            depth_knn = F.interpolate(depth_knn_m_dwon, size=(H, W), mode="bicubic", align_corners=False)
            depth_cspn_m = (depth_cspn_m + depth_knn) / 2
        elif diffusion == '2D-3D':
            B, C, H, W = sparse.shape
            feat = feat.unsqueeze(1)

            point_homo, point = self.depth2point(depth_polyfit, K, cam=cam)
            point_down = F.interpolate(point, size=(H//self.scale, W//self.scale), mode="bicubic", align_corners=False)        
            cost_feat_down = F.interpolate(feat, size=(9, H//self.scale, W//self.scale), mode="trilinear", align_corners=False).squeeze(1)
            cost_feat_down = torch.cat((point_down, cost_feat_down), dim=1)
            ip_median_down = F.interpolate(ip_median, size=(H//self.scale, W//self.scale), mode="bicubic", align_corners=False)
            sparse_down = self.sds(sparse)
            depth_cspn_m_dwon = self.CSPN_down(sparse_down, ip_median_down.clone(), cost_feat_down).detach()
            depth_cspn_m_dwon = F.interpolate(depth_cspn_m_dwon, size=(H, W), mode="bicubic", align_corners=False)
            cost_feat = F.interpolate(feat, size=(16, H, W), mode="trilinear", align_corners=False).squeeze(1)
            depth_cspn_m = self.CSPN(sparse, depth_cspn_m_dwon, cost_feat).detach()
            depth_cspn_m_dwon = F.interpolate(depth_cspn_m, size=(H//self.scale, W//self.scale), mode="bicubic", align_corners=False)
            depth_knn_m_dwon = self.knnv1(cost_feat_down, point_down, depth_cspn_m_dwon.clone(), sparse_down, data).detach()
            depth_cspn_m = F.interpolate(depth_knn_m_dwon, size=(H, W), mode="bicubic", align_corners=False)
              
        return depth_cspn_m

    def _add(self, a, b):
        _, _, Hd, Wd = a.shape
        _, _, He, We = b.shape

        if Hd==He and Wd==We:
            return a+b

        a = F.interpolate(a, size=(He, We), mode='bilinear', align_corners=False)
        return a+b

    def _multiply(self, a, b):
        _, _, Hd, Wd = a.shape
        _, _, He, We = b.shape

        if Hd==He and Wd==We:
            return a*b

        a = F.interpolate(a, size=(He, We), mode='bilinear', align_corners=False)
        return a*b

    def forward(self, inputs, epoch, is_test, data):
        rgb = inputs['rgb']
        rgb_mde = inputs['rgb_mde']
        sparse = inputs['sparse']
        ip = inputs['ip']
        K = inputs['K']

        # b, 1, 384, 384; 1/2, 1/4, 1/8, 1/16; 1/4, 1/8, 1/16, 1/32
        depth_mde, path_feats, layer_feats = self.mde(inputs)
        depth_mde = F.interpolate(depth_mde, size=sparse.shape[2:], mode="bicubic", align_corners=False)
        if data == 'NK': # N=0, K=1
            depth_polyfit = self.align_inverse_depth_nk(depth_mde.clone(), sparse, inputs['dataset'])
            ip_median = self.align_depth_nk(ip.clone(), sparse, inputs['dataset'])
        else:
            depth_polyfit = self.align_inverse_depth(depth_mde.clone(), sparse, data)
            ip_median = self.align_depth(ip.clone(), sparse, data)
        depth_polyfit = depth_polyfit.detach()
        ip_median = ip_median.detach()

        B, _, H, W = sparse.shape

        # diffusion
        depth_diff = self.dual_diffusion(path_feats[0], ip_median, depth_polyfit, sparse, inputs, data)
        
        # residual
        sparse_residual = (sparse - depth_diff) * (sparse>0)
        residual, confidence_residual = self.rb(sparse, depth_diff, sparse_residual, path_feats)
        depth_residual = self._add(residual, depth_diff)
        
        # bin range
        if epoch > 39:
            resi = residual
            conf = confidence_residual
            depr = depth_residual
        else:
            resi = residual.detach()
            conf = confidence_residual.detach()
            depr = depth_residual.detach()

        width = resi.abs() * (2. - conf) + 0.05
        max_bin = (1. + self.alpha1) * width + self.beta1
        max_bin = torch.clamp(max_bin, min=0.05)
        min_bin = (-1. + self.alpha2) * width + self.beta2
        min_bin = torch.clamp(min_bin, max=-0.05)
        step = (max_bin - min_bin) / (self.num - 1)
        bins = torch.arange(1.0, self.num-1, 1).reshape(1, -1, 1, 1).to(sparse.device)
        bins = min_bin + step * bins
        bins = torch.cat((min_bin, bins, max_bin), dim=1) 
 
        sparse_residual = (sparse - depr) * (sparse>0)
        laplace = F.softmax(-torch.abs(sparse_residual - bins), dim=1)
        laplace[(sparse <= 0).repeat(1, self.num, 1, 1)] = 0.0
        laplace = laplace.detach()

        # image branch
        similarity, confidence_img, feat = self.ib(sparse, depr, sparse_residual, laplace, conf, path_feats) # 1/2 1/4 1/8 1/16 1/32 1/64
        similarity = F.interpolate(similarity, size=(H, W), mode="bilinear", align_corners=False)
        similarity = F.softmax(similarity, dim=1)
        confidence_img = F.interpolate(confidence_img, size=(H, W), mode="bilinear", align_corners=False)
        confidence_img = F.sigmoid(confidence_img)
        residual_img = torch.sum(self._multiply(bins, similarity), 1, keepdim=True)
        depth_img = self._add(residual_img, depth_residual)

        # pixel branch
        if epoch > 39:
            depi = depth_img
            conf = confidence_img
        else:
            depi = depth_img.detach()
            conf = confidence_img.detach()
       
        sparse_residual = (sparse - depi) * (sparse>0)
        global_offset, global_conf = self.pb(sparse, depi, sparse_residual, conf, feat)
        
        bins_pix = global_offset
        if epoch > 39:
            similarity1 = similarity
        else:
            similarity1 = similarity.detach()
        residual_pix = torch.sum(self._multiply(bins_pix, similarity1), 1, keepdim=True) 
        depth_pix = self._add(residual_pix, depth_img)

        if self.training: 
            outputs = {'depth': [depth_residual, depth_img, depth_pix], \
                        'confidence': [confidence_residual, confidence_img, global_conf]}
        else:
            # post-processing
            depth_post = self.post_processing(feat, depth_pix, sparse, data)
            outputs = {'depth': [ip, depth_polyfit, depth_diff, depth_residual, depth_img, depth_pix, depth_post]}

        if self.training:         
            losses = {}
            loss = 0
            for loss_function, loss_weight in self.losses.items():
                loss_type = loss_function.__name__
                losses[loss_type] = loss_function(inputs, outputs, epoch) * loss_weight
            for loss_type, value in losses.items():
                to_optim = value.mean()
                loss += to_optim
            losses["loss"] = loss
            return losses, outputs 
        else:
            if is_test is not True:
                errors = self.compute_depth_error(inputs, outputs, data)
                return errors, outputs
            else:
                return outputs

    def compute_confidence_loss(self, inputs, outputs, epoch):
        target = inputs['target']
        mask = inputs['target'] > 0
        key = 'confidence'
        total_loss = 0

        if epoch < 2:
            weights = [0.1, 0.1, 0.1]
        elif epoch < 10:
            weights = [1.0, 1.0, 1.0]
        elif epoch < 15:
            weights = [0.5, 0.75, 0.75]
        elif epoch < 20:
            weights = [0.25, 0.5, 0.5]
        elif epoch < 25:
            weights = [0.1, 0.25, 0.25]
        else:
            weights = [0.05, 0.1, 0.1]

        for idx, (conf, depth, w) in enumerate(zip(outputs[key], outputs['depth'], weights)):
            item1 = (target - depth.detach()).abs()
            item2 = (target + depth.detach() + 1e-7)
            conf_gt = torch.exp(-5 * item1 / item2) # 置信度

            l1_loss = F.l1_loss(conf[mask], conf_gt[mask], reduction='none')
            total_loss = total_loss + l1_loss.mean() * w

        return total_loss

    def compute_depth_loss(self, inputs, outputs, epoch):
        target = inputs['target']
        mask = inputs['target'] > 0
        key = 'depth'
        total_loss = 0

        if epoch < 10:
            weights = [1.0, 1.0, 1.0]
        elif epoch < 15:
            weights = [0.5, 1.0, 1.0]
        elif epoch < 20:
            weights = [0.5, 0.5, 1.0]
        elif epoch < 25:
            weights = [0.1, 0.5, 1.0]
        else:
            weights = [0.1, 0.25, 1.0]

        for idx, (pred, w) in enumerate(zip(outputs[key], weights)):
            l1_loss = F.l1_loss(pred[mask], target[mask], reduction='none')
            # mse_loss = F.mse_loss(pred[mask], target[mask], reduction='none')
            total_loss = total_loss + (l1_loss.mean()) * w
        return total_loss

    def compute_depth_error(self, inputs, outputs, data):
        errors = {'mae': [], 'rmse': [], 'imae': [], 'irmse': [],
                'rel': [], 'd1': [], 'd2': [], 'd3': []}

        gt = inputs['target']

        if data == self.VOID1500.name:
            mask = (gt > 0.2) & (gt < 5)
        elif data == self.DIODEo.name:
            mask = (gt > 1) & (gt < 250)
        elif data == self.VKITTI2.name:
            mask = (gt > 1) & (gt < 250)
        elif data == self.Cityscape.name:
            mask = (gt > 1) & (gt < 250)
        elif data == self.TOFDC.name:
            mask = (gt > 0) & (gt < 6)
        else:
            mask = gt > 0

        for idx, depth in enumerate(outputs['depth']):
            depth = depth[mask]
            target = gt[mask]

            err = 1000.0 * depth - 1000.0 * target # 
            errors['mae'].append(err.abs().mean())          
            errors['rmse'].append((err ** 2).mean().sqrt()) 
            
            err = 1.0 / (0.001 * depth) - 1.0 / (0.001 * target) 
            errors['imae'].append(err.abs().mean())
            errors['irmse'].append((err ** 2).mean().sqrt())

            diff_mat = torch.abs(target-depth)
            rel_mat = torch.div(diff_mat, target)
            rel = rel_mat.mean()
            y_over_z = torch.div(target, depth)
            z_over_y = torch.div(depth, target)
            max_ratio = torch.max(y_over_z, z_over_y)
            d1 = torch.sum(max_ratio < 1.25) / float(mask.sum())
            d2 = torch.sum(max_ratio < 1.25**2) / float(mask.sum())
            d3 = torch.sum(max_ratio < 1.25**3) / float(mask.sum())
            errors['rel'].append(rel*100.0)
            errors['d1'].append(d1*100.0)
            errors['d2'].append(d2*100.0)
            errors['d3'].append(d3*100.0)

        return errors

if __name__ == "__main__":
    import argparse
    import os
    import os.path as osp
    from omegaconf import OmegaConf

    def load_config(cfg_file):
        cfg = OmegaConf.load(cfg_file)
        if '_base_' in cfg:
            if isinstance(cfg._base_, str):
                base_cfg = OmegaConf.load(osp.join(osp.dirname(cfg_file), cfg._base_))
            else:
                base_cfg = OmegaConf.merge(OmegaConf.load(f) for f in cfg._base_)
            cfg = OmegaConf.merge(base_cfg, cfg)
        return cfg

    def get_config(args):
        cfg = load_config(args.cfg)
        OmegaConf.set_struct(cfg, True)
        OmegaConf.set_readonly(cfg, True)
        return cfg

    parser = argparse.ArgumentParser(description='DCNet')
    parser.add_argument("--cfg", type=str, default='/data/slchen/paper-large/DC/configs/nyu.yml')
    args = parser.parse_args()
    cfg = get_config(args)

    inputs = {}
    inputs['rgb'] = torch.rand(2, 3, 480, 640)
    inputs['rgb_r'] = torch.rand(2, 3, 384, 384)
    inputs['sparse'] = torch.rand(2, 1, 480, 640)
    inputs['target'] = torch.rand(2, 1, 480, 640)

    dc = DCNet(cfg)

    dc(inputs, 1, True)