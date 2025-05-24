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
# from .trans_metric3d import transform_data_scalecano

class StanfordDataset(data.Dataset):
    def __init__(self, cfg, mode='train'):
        super(StanfordDataset, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.max = cfg.Stanford.max
        self.valrgb = cfg.Stanford.valrgb
        self.valgt = cfg.Stanford.valgt
        self.valraw = cfg.Stanford.valraw
        
        self.resize_height = cfg.Stanford.resize_height
        self.resize_width = cfg.Stanford.resize_width
        # self.valk = cfg.Stanford.valk
        self.K = [540.0, 0.0, 540.0, 0.0, 540.0, 540.0, 0.0, 0.0, 1.0]
        
        self.resize_size_midas = (cfg.MDEBranch.midas.resize_h, cfg.MDEBranch.midas.resize_w)
        
        if mode== 'train':
            pass
        elif mode== 'val':
            self.rgb_files, self.sparse_files, self.target_files = self.load_path(cfg.Stanford.vallist)

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
            
            self.data_basic = cfg.SCANNET.data_basic
        
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
        return rgb_files, sparse_files, target_files

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

            rgb = self.get_rgb(rgb_file)
            sparse = self.get_depth(sparse_file) 
            target = self.get_depth(target_file)
            K = np.reshape(self.K, [3, 3])

            rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() # b,c,h,w
            sparse = torch.from_numpy(sparse).permute(2, 0, 1).unsqueeze(0).float()
            target = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float()
            height = rgb.shape[2]
            width = rgb.shape[3]

            rgb = F.interpolate(rgb, size=(self.resize_height, self.resize_width), mode='bilinear', align_corners=True)
            sparse = F.interpolate(sparse, size=(self.resize_height, self.resize_width), mode='nearest')
            target = F.interpolate(target, size=(self.resize_height, self.resize_width), mode='bilinear', align_corners=True)

            K[0][0] = K[0][0]*rgb.shape[2]/width
            K[1][1] = K[1][1]*rgb.shape[2]/height
            K[0][2] = K[0][2]*rgb.shape[2]/width
            K[1][2] = K[1][2]*rgb.shape[2]/height

            rgb = rgb.squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
            sparse = sparse.squeeze(0).permute(1,2,0).numpy()
            target = target.squeeze(0).permute(1,2,0).numpy()

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

            inputs['rgb'] = rgb
            inputs['rgb_s'] = rgb_s
            inputs['rgb_mde'] = rgb_mde
            inputs['sparse'] = self.trans_tensor(sparse)
            inputs['target'] = self.trans_tensor(target)
            inputs['K'] = self.trans_tensor(K).squeeze(0)

            sparse_ip2 = ip(inputs['sparse'].clone().numpy().squeeze(0), max_depth=inputs['sparse'].max())
            inputs['ip'] = self.trans_tensor(sparse_ip2)

        return inputs

    def get_rgb(self, file):
        assert os.path.exists(file), "file not found: {}".format(file)
        img_file = Image.open(file)
        rgb_png = np.array(img_file.copy(), dtype='uint8')
        img_file.close()
        return rgb_png

    def get_depth(self, file):
        assert os.path.exists(file), "file not found: {}".format(file)
        img_file = Image.open(file)
        depth_png = np.array(img_file, dtype=int)
        img_file.close()

        depth = depth_png.astype(np.float32) / 256.
        depth[depth <= 0] = 0.0
        depth = np.expand_dims(depth, -1)
        return depth
    
    def get_knn_candidate(self, sparse, num_sample):
        channel, height, width = sparse.shape

        assert channel == 1

        idx_nnz = torch.nonzero(sparse.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        return idx_nnz.squeeze()
    