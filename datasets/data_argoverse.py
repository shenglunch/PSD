import os
import random
import math
import torch
import torch.utils.data as data
import PIL.Image as pil
import numpy as np
from PIL import Image
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F

from datasets import transforms
from .ip_basic import fill_in_fast as ip
from .trans_metric3d import transform_data_scalecano

scene01 = [3771.11, 0.000000e+00, 1140.42, \
    0.000000e+00, 3771.11, 1047.26, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene02 = [3847.63, 0.000000e+00, 1298.50, \
    0.000000e+00, 3847.63, 1058.25, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene03 = [3784.85, 0.000000e+00, 1145.47, \
    0.000000e+00, 3784.85, 1034.68, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene04 = [3757.45, 0.000000e+00, 1269.16, \
    0.000000e+00, 3757.45, 1044.85, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene05 = [3727.40, 0.000000e+00, 1162.27, \
    0.000000e+00, 3727.40, 1041.31, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene06 = [3742.00, 0.000000e+00, 1238.78, \
    0.000000e+00, 3742.00, 1051.03, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene07 = [3784.95, 0.000000e+00, 1212.16, \
    0.000000e+00, 3784.95, 1066.88, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene08 = [3756.49, 0.000000e+00, 1224.17, \
    0.000000e+00, 3756.49, 1047.55, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene09 = [3779.20, 0.000000e+00, 1115.40, \
    0.000000e+00, 3779.20, 1015.92, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene10 = [3742.00, 0.000000e+00, 1238.78, \
    0.000000e+00, 3742.00, 1051.03, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene11 = [3741.13, 0.000000e+00, 1046.63, \
    0.000000e+00, 3741.13, 1037.92, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene12 = [3735.65, 0.000000e+00, 1090.51, \
    0.000000e+00, 3735.65, 1031.38, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene13 = [3722.72, 0.000000e+00, 1241.33, \
    0.000000e+00, 3722.72, 1046.56, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene14 = [3784.95, 0.000000e+00, 1212.16, \
    0.000000e+00, 3784.95, 1066.88, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene15 = [3771.11, 0.000000e+00, 1140.42, \
    0.000000e+00, 3771.11, 1047.26, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene16 = [3757.45, 0.000000e+00, 1269.16, \
    0.000000e+00, 3757.45, 1044.85, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
scene17 = [3757.45, 0.000000e+00, 1269.16, \
    0.000000e+00, 3757.45, 1044.85, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]

Ks = {'scene01': scene01, 'scene02': scene02, 'scene03': scene03, 'scene04': scene04, 'scene05': scene05,
        'scene06': scene06, 'scene07': scene07, 'scene08': scene08, 'scene09': scene09, 'scene10': scene10,
        'scene11': scene11, 'scene12': scene12, 'scene13': scene13, 'scene14': scene14, 'scene15': scene15,
        'scene16': scene16, 'scene17': scene17}

def val_transform(rgb, target, K, cfg):
    h, w = rgb.shape[0], rgb.shape[1]
    
    # transforms_list = [
    #     transforms.BottomCrop((cfg.ARGOVERSE.bottom_height, cfg.ARGOVERSE.bottom_width))]

    # transform_geometric = transforms.Compose(transforms_list)

    # rgb = transform_geometric(rgb)
    # target = transform_geometric(target)

    y_start = (h - cfg.ARGOVERSE.bottom_height) # BottomCrop!!!
    x_start = (w - cfg.ARGOVERSE.bottom_width) // 2
    K = K + [[0.0, 0.0, -x_start],
             [0.0, 0.0, -y_start],
             [0.0, 0.0, 0.0]]
            
    return rgb, target, K

class ARGOVERSEDataset(data.Dataset):
    def __init__(self, cfg, mode='val'):
        super(ARGOVERSEDataset, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.rate = cfg.ARGOVERSE.rate
        self.max = cfg.ARGOVERSE.max
        self.valrgb = cfg.ARGOVERSE.valrgb
        self.valgt = cfg.ARGOVERSE.valgt
        self.valraw = cfg.ARGOVERSE.valraw
        self.valk = cfg.ARGOVERSE.valk
        
        if mode== 'train':
            pass 
        elif mode== 'val':
            self.rgb_files, self.sparse_files, self.target_files, self.intrinsic_files  = self.load_path(cfg.ARGOVERSE.vallist)

        if cfg.MDEBranch.backbone == 'midas':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.NormalizeTensor(cfg.MDEBranch.midas.mean, cfg.MDEBranch.midas.std),
                transforms.ConvertFloat()])
            self.trans_mde = transforms.Compose([
                transforms.ResizeV2((cfg.MDEBranch.midas.resize_h, cfg.MDEBranch.midas.resize_w), 1)])
        
        elif cfg.MDEBranch.backbone == 'depthanything' or cfg.MDEBranch.backbone == 'depthanythingv2':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.NormalizeTensor(cfg.MDEBranch.depany.mean, cfg.MDEBranch.depany.std),
                transforms.ConvertFloat()])

            self.trans_mde = transforms.Compose([
                transforms.ResizeV4(
                width=cfg.MDEBranch.depany.resize_w, 
                height=cfg.MDEBranch.depany.resize_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=3,)])
                
        elif cfg.MDEBranch.backbone == 'depthpro':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.NormalizeTensor(cfg.MDEBranch.depthpro.mean, cfg.MDEBranch.depthpro.std),
                transforms.ConvertFloat()])

            self.trans_mde = transforms.Compose([
                transforms.ResizeV2((cfg.MDEBranch.depthpro.resize_h, cfg.MDEBranch.depthpro.resize_w), 1)])

        elif cfg.MDEBranch.backbone == 'metricv2':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.NormalizeTensor(cfg.MDEBranch.metricv2.mean, cfg.MDEBranch.metricv2.std),
                transforms.ConvertFloat()])
            
            self.data_basic = cfg.ARGOVERSE.data_basic
        
        elif cfg.MDEBranch.backbone == 'promptda':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.ConvertFloat()])

            self.trans_mde = transforms.Compose([
                transforms.ResizeV4(
                width=cfg.MDEBranch.promptda.resize_w, 
                height=cfg.MDEBranch.promptda.resize_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=3,)])
        
        elif cfg.MDEBranch.backbone == 'unidepth':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.ConvertFloat()])

            self.trans_mde = transforms.Compose([
                transforms.ResizeV4(
                width=cfg.MDEBranch.unidepth.resize_w, 
                height=cfg.MDEBranch.unidepth.resize_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=3,)])

        self.backbone = cfg.MDEBranch.backbone

        self.trans_tensor = transforms.ToTensor()

        self.trans_rgb_nonnorm = transforms.Compose([
                transforms.ToTensor(),
                transforms.ConvertFloat()])

    def load_path(self, list_filename):
        assert os.path.exists(list_filename), "file not found: {}".format(list_filename)
        with open(list_filename, "r") as f:
            lines = [line.rstrip() for line in f.readlines()]

        splits = [line.split() for line in lines]
        rgb_files = [x[0] for x in splits]
        sparse_files = [x[1] for x in splits]
        target_files = [x[2] for x in splits]
        intrinsic_files = [x[3] for x in splits]
        return rgb_files, sparse_files, target_files, intrinsic_files

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, index):
        inputs={}
        
        if self.mode == 'train':
            pass 
        elif self.mode == 'val':
            rgb_file = os.path.join(self.valrgb, self.rgb_files[index])
            sparse_file = os.path.join(self.valraw, self.sparse_files[index])
            target_file = os.path.join(self.valgt, self.target_files[index])
            intrinsic_file = os.path.join(self.valk, self.intrinsic_files[index])
            
            rgb = self.get_rgb(rgb_file)
            sparse = self.get_depth(sparse_file) 
            target = self.get_depth(target_file)
            K = np.reshape(np.loadtxt(intrinsic_file, dtype=np.float32), [3, 3])

            h, w, c = rgb.shape

            if self.backbone == 'metricv2':
                intrinsic = [K[0][0], K[1][1], K[0][2], K[1][2]]
                rgb_mde, cam_models_stacks, pad, label_scale_factor = transform_data_scalecano(rgb, intrinsic, self.data_basic)     
                inputs['cam_model'] = cam_models_stacks
                inputs['pad_info'] = pad
                inputs['scale_info'] = label_scale_factor
                inputs['normalize_scale'] = self.data_basic.depth_range[1]
            else:
                rgb_mde = self.trans_rgb2tensor(self.trans_mde(rgb))
            rgb_s = self.trans_rgb_nonnorm(rgb)
            rgb = self.trans_rgb2tensor(rgb)

            inputs['target'] = self.trans_tensor(target)
            inputs['sparse'] = self.trans_tensor(sparse)
            inputs['K'] = self.trans_tensor(K).squeeze(0)
            inputs['rgb'] = rgb
            inputs['rgb_s'] = rgb_s
            inputs['rgb_mde'] = rgb_mde

            sparse_ip2 = ip(inputs['sparse'].clone().numpy().squeeze(0), max_depth=inputs['sparse'].max())
            inputs['ip'] = self.trans_tensor(sparse_ip2)
        return inputs

    def get_rgb(self, file):
        assert os.path.exists(file), "file not found: {}".format(file)
        img_file = Image.open(file)
        rgb_png = np.array(img_file.copy(), dtype='uint8')
        img_file.close()
        return rgb_png

    def get_disp(self, filename):
        data, scale = readPFM(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data
    
    def get_depth(self, file):
        assert os.path.exists(file), "file not found: {}".format(file)
        img_file = Image.open(file)
        depth_png = np.array(img_file, dtype=int)
        img_file.close()

        depth = depth_png.astype(np.float32) / 256.
        depth[depth <= 0] = 0.0
        depth = np.expand_dims(depth, -1)
        return depth