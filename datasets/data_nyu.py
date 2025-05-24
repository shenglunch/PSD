
import os
import warnings
import random
import numpy as np
import h5py
import torch.utils.data as data
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import cv2
warnings.filterwarnings("ignore", category=UserWarning)

from datasets import transforms
from .ip_basic import fill_in_fast as ip
from .pseudo_hole import SegmentationHighLight, Spatter
from .trans_metric3d import transform_data_scalecano

K_nyu = [
    5.1885790117450188e+02 / 2.0, 0.0, 3.2558244941119034e+02 / 2.0 - 8.0, \
    0.0, 5.1946961112127485e+02 / 2.0, 2.5373616633400465e+02 / 2.0 - 6.0, \
    0.0, 0.0, 1.0]

class NYUDataset(data.Dataset):
    def __init__(self, cfg, mode='train'):
        super(NYUDataset, self).__init__()

        self.cfg = cfg
        self.mode = mode
        self.path = cfg.NYUv2.path
        self.rate = cfg.NYUv2.rate
        self.max = cfg.NYUv2.max

        self.valrgb = cfg.NYUv2.valrgb
        self.valgt = cfg.NYUv2.valgt
        self.valraw = cfg.NYUv2.valraw

        self.setting = cfg.train.dataset.img_aug
        self.jitter = self.setting.color_jitter
        self.degree = self.setting.degree
        self.scale = self.setting.scale

        self.height = cfg.NYUv2.height
        self.crop_size = (cfg.NYUv2.crop_height, cfg.NYUv2.crop_width)
        self.h = cfg.NYUv2.crop_height
        self.w = cfg.NYUv2.crop_width


        if mode== 'train':
            self.sample_list = self.load_path(cfg.NYUv2.trainlist)
        elif mode== 'val':
            self.rgb_files, self.sparse_files, self.target_files = self.load_path(cfg.NYUv2.vallist)

        if cfg.MDEBranch.backbone == 'midas':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.NormalizeTensor(cfg.MDEBranch.midas.mean, cfg.MDEBranch.midas.std),
                transforms.ConvertFloat()])
            self.trans_mde = transforms.Compose([
                transforms.ResizeV1((cfg.MDEBranch.midas.resize_h, cfg.MDEBranch.midas.resize_w), Image.BILINEAR)])
            self.trans_mde2 = transforms.Compose([
                transforms.ResizeV2((cfg.MDEBranch.midas.resize_h, cfg.MDEBranch.midas.resize_w), Image.BILINEAR)])
        
        elif cfg.MDEBranch.backbone == 'depthanything' or cfg.MDEBranch.backbone == 'depthanythingv2':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.NormalizeTensor(cfg.MDEBranch.depany.mean, cfg.MDEBranch.depany.std),
                transforms.ConvertFloat()])

            self.trans_mde = transforms.Compose([
                transforms.ResizeV3(
                width=cfg.MDEBranch.depany.resize_w, 
                height=cfg.MDEBranch.depany.resize_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=Image.BICUBIC,)])
            
            self.trans_mde2 = transforms.Compose([
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
                transforms.ResizeV1((cfg.MDEBranch.depthpro.resize_h, cfg.MDEBranch.depthpro.resize_w), Image.BILINEAR)])
            self.trans_mde2 = transforms.Compose([
                transforms.ResizeV2((cfg.MDEBranch.depthpro.resize_h, cfg.MDEBranch.depthpro.resize_w), Image.BILINEAR)])
        
        elif cfg.MDEBranch.backbone == 'metricv2':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.NormalizeTensor(cfg.MDEBranch.metricv2.mean, cfg.MDEBranch.metricv2.std),
                transforms.ConvertFloat()])
            
            self.data_basic = cfg.NYUv2.data_basic
        
        elif cfg.MDEBranch.backbone == 'promptda':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.ConvertFloat()])

            self.trans_mde = transforms.Compose([
                transforms.ResizeV3(
                width=cfg.MDEBranch.promptda.resize_w, 
                height=cfg.MDEBranch.promptda.resize_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=Image.BICUBIC,)])
            
            self.trans_mde2 = transforms.Compose([
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
                transforms.ResizeV3(
                width=cfg.MDEBranch.unidepth.resize_w, 
                height=cfg.MDEBranch.unidepth.resize_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=Image.BICUBIC,)])
            
            self.trans_mde2 = transforms.Compose([
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

        if self.mode == 'train':
            with open(list_filename, "r") as f:
                sample = [os.path.join(self.path, line.rstrip()) for line in f.readlines()] 
        
            return sample
        
        elif self.mode == 'val':
            with open(list_filename, "r") as f:
                lines = [line.rstrip() for line in f.readlines()]

            splits = [line.split() for line in lines]
            rgb_files = [x[0] for x in splits]
            sparse_files = [x[1] for x in splits]
            target_files = [x[2] for x in splits]
            return rgb_files, sparse_files, target_files

    def __len__(self):
        if self.mode == 'train':
            return len(self.sample_list)
        else:
            return len(self.rgb_files)

    def __getitem__(self, index):
        inputs={}
        K = np.reshape(K_nyu, [3, 3])

        if self.mode == 'train':
            path_file = self.sample_list[index]

            f = h5py.File(path_file, 'r')
            rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
            dep_h5 = f['depth'][:]

            rgb = Image.fromarray(rgb_h5, mode='RGB')
            target = Image.fromarray(dep_h5.astype('float32'), mode='F')

            cur_scale = np.random.uniform(1.0, self.scale)
            if index == 24132:
                cur_scale = 1
            height_scale = int(self.height * cur_scale)

            do_flip = np.random.uniform(0.0, 1.0) > 0.5
            if do_flip:
                rgb = transforms.hflip(rgb)
                target = transforms.hflip(target)

            degree = np.random.uniform(-self.degree, self.degree)
            rgb = transforms.rotate(rgb, degree, Image.BILINEAR)
            target = transforms.rotate(target, degree, Image.NEAREST)

            rgb_hole = rgb
            target_hole = target

            trans_rgb = transforms.Compose([
                transforms.ResizeV1(height_scale, Image.BILINEAR),
                transforms.ColorJitter(self.jitter),
                transforms.CenterCrop(self.crop_size),])
            
            trans_dep = transforms.Compose([
                transforms.ResizeV1(height_scale, Image.NEAREST),
                transforms.CenterCrop(self.crop_size),
                transforms.ToNumpy(),
                transforms.ToTensor(),
                transforms.ConvertFloat(),])

            rgb = trans_rgb(rgb)
            if self.backbone == 'metricv2':
                intrinsic = [K[0][0], K[1][1], K[0][2], K[1][2]]
                rgb_mde, cam_models_stacks, pad, label_scale_factor = transform_data_scalecano(rgb, intrinsic, self.data_basic)     
                inputs['cam_model'] = cam_models_stacks
                inputs['pad_info'] = pad
                inputs['scale_info'] = label_scale_factor
                inputs['normalize_scale'] = self.data_basic.depth_range[1]
            else:
                rgb_mde = self.trans_rgb2tensor(self.trans_mde(rgb))
            rgb = self.trans_rgb2tensor(rgb)

            target = trans_dep(target)
            target = target / cur_scale

            K[0][0] = K[0][0] * cur_scale
            K[1][1] = K[1][1] * cur_scale

        else:
            rgb_file = os.path.join(self.valrgb, self.rgb_files[index])
            sparse_file = os.path.join(self.valraw, self.sparse_files[index])
            target_file = os.path.join(self.valgt, self.target_files[index])

            rgb = self.get_rgb(rgb_file)
            sparse = self.get_depth(sparse_file) 
            target = self.get_depth(target_file)

            trans_dep = transforms.Compose([
                transforms.ToTensor(),
                transforms.ConvertFloat(),])

            if self.backbone == 'metricv2':
                intrinsic = [K[0][0], K[1][1], K[0][2], K[1][2]]
                rgb_mde, cam_models_stacks, pad, label_scale_factor = transform_data_scalecano(rgb, intrinsic, self.data_basic)     
                inputs['cam_model'] = cam_models_stacks
                inputs['pad_info'] = pad
                inputs['scale_info'] = label_scale_factor
                inputs['normalize_scale'] = self.data_basic.depth_range[1]
            else:
                rgb_mde = self.trans_rgb2tensor(self.trans_mde2(rgb))

            rgb = self.trans_rgb2tensor(rgb)

            target = trans_dep(target)
            sparse = trans_dep(sparse)

        inputs['rgb'] = rgb
        inputs['rgb_mde'] = rgb_mde
        inputs['target'] = target

        if self.mode == 'train':
            do_hole = np.random.uniform(0.0, 1.0) < 0.2
            if do_hole:
                raw_depth = target_hole.copy()
                masks = []
                pseudo_sample = {'rgb': rgb_hole ,'raw_depth': raw_depth }
                seg_t = SegmentationHighLight()
                masks.append(seg_t(sample=pseudo_sample))
                spatter_t = Spatter()
                masks.append(spatter_t(sample=pseudo_sample))

                # combine all pseudo masks
                pseudo_maks = np.zeros_like(raw_depth, dtype=bool)

                for m in masks:
                    pseudo_maks |= m
                
                pseudo_depth = np.array(raw_depth)
                pseudo_depth[pseudo_maks] = 0.0
                pseudo_depth = Image.fromarray(pseudo_depth, mode='F')
                pseudo_depth = trans_dep(pseudo_depth)
                pseudo_depth = pseudo_depth / cur_scale
                mask = pseudo_depth > 0.000001
            else:
                do_sparse = np.random.uniform(0.0, 1.0) < 0.1
                if do_sparse:
                    num = 400
                else:
                    num = int(self.rate*self.h*self.w)
                sparse, _ = self.get_sparse_depth(inputs['target'].clone(), num)
                mask = sparse > 0.000001

            sparse = target.clone()
            do_noiseblur = np.random.uniform(0.0, 1.0) < 0.20
            if do_noiseblur:
                sparse_noise = self._noiseblur(sparse)
            else:
                sparse_noise = sparse
            inputs['sparse'] = sparse_noise * mask
        else:
            inputs['sparse'] = sparse

        inputs['K'] = torch.from_numpy(K)

        sparse_ip2 = ip(inputs['sparse'].clone().numpy().squeeze(0), max_depth=inputs['sparse'].max())
        inputs['ip'] = self.trans_tensor(sparse_ip2)

        return inputs

    def _noiseblur(self, raw, p_noise=0.5, p_blur=0.5):
        raw_shape = raw.shape
        raw_min = raw.min()
        raw_max = raw.max()
        # add noise
        if np.random.uniform(0.0, 1.0) < p_noise:
            gaussian_noise = torch.ones(raw_shape).normal_(0, np.random.uniform(0.01, 0.1))
            gaussian_noise = self._sample(gaussian_noise, np.random.uniform(0.25, 0.75))
            raw = raw + gaussian_noise
        # add blur
        if np.random.uniform(0.0, 1.0) < p_blur:
            sample_factor = 2 ** (np.random.randint(1, 4))
            import torchvision.transforms as trans
            blur_trans = trans.Compose(
                [
                    trans.Resize(
                        (raw_shape[1] // sample_factor, raw_shape[2] // sample_factor),
                        interpolation=Image.NEAREST, #TF.InterpolationMode.NEAREST,
                        # antialias=True,
                    ),
                    trans.Resize(
                        (raw_shape[1], raw_shape[2]),
                        interpolation=Image.NEAREST, #TF.InterpolationMode.NEAREST,
                        # antialias=True,
                    ),
                ]
            )
            raw = blur_trans(raw)
        return torch.clamp(raw, raw_min, raw_max)
    
    def _sample(self, raw, zero_rate):
        data_shape = raw.shape
        if zero_rate == 0.0:
            return raw
        elif zero_rate == 1.0:
            return torch.zeros(data_shape)
        else:
            random_point = torch.ones(data_shape).uniform_(0.0, 1.0)
            random_point[random_point <= zero_rate] = 0.0
            random_point[random_point > zero_rate] = 1.0
            return raw * random_point
    
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
        depth = np.expand_dims(depth, -1) # 深度单位：米
        return depth

    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp, idx_nnz.squeeze()
    