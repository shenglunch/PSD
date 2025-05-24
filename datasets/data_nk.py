import os
import random
import math
import torch
import torch.utils.data as data
import h5py
import PIL.Image as pil
import numpy as np
from PIL import Image
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F

from datasets import transforms
from .ip_basic import fill_in_fast as ip
from .pseudo_hole import SegmentationHighLight, Spatter
from .trans_metric3d import transform_data_scalecano

# 2011_09_26
K_26_02 = [7.215377e+02, 0.000000e+00, 6.095593e+02, \
    0.000000e+00, 7.215377e+02, 1.728540e+02, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
K_26_03 = [7.215377e+02, 0.000000e+00, 6.095593e+02, \
    0.000000e+00, 7.215377e+02, 1.728540e+02, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
# 2011_09_28
K_28_02 = [7.070493e+02, 0.000000e+00, 6.040814e+02, \
    0.000000e+00, 7.070493e+02, 1.805066e+02, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
K_28_03 = [7.070493e+02, 0.000000e+00, 6.040814e+02, \
    0.000000e+00, 7.070493e+02, 1.805066e+02, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
# 2011_09_29
K_29_02 = [7.183351e+02, 0.000000e+00, 6.003891e+02, \
    0.000000e+00, 7.183351e+02, 1.815122e+02, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
K_29_03 = [7.183351e+02, 0.000000e+00, 6.003891e+02, \
    0.000000e+00, 7.183351e+02, 1.815122e+02, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
# 2011_09_30
K_30_02 = [7.070912e+02, 0.000000e+00, 6.018873e+02, \
    0.000000e+00, 7.070912e+02, 1.831104e+02, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
K_30_03 = [7.070912e+02, 0.000000e+00, 6.018873e+02, \
    0.000000e+00, 7.070912e+02, 1.831104e+02, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
# 2011_10_03
K_03_02 = [7.188560e+02, 0.000000e+00, 6.071928e+02, \
    0.000000e+00, 7.188560e+02, 1.852157e+02, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
K_03_03 = [7.188560e+02, 0.000000e+00, 6.071928e+02, \
    0.000000e+00, 7.188560e+02, 1.852157e+02, \
    0.000000e+00, 0.000000e+00, 1.000000e+00]
Ks = {'2011_09_26': K_26_02, '2011_09_28': K_28_02, '2011_09_29': K_29_02, '2011_09_30': K_30_02, '2011_10_03': K_03_02}

K_nyu = [
    5.1885790117450188e+02 / 2.0, 0.0, 3.2558244941119034e+02 / 2.0 - 8.0, \
    0.0, 5.1946961112127485e+02 / 2.0, 2.5373616633400465e+02 / 2.0 - 6.0, \
    0.0, 0.0, 1.0]

def kitti_train_transform(rgb, sparse, target, K, cfg):
    setting = cfg.train.dataset.img_aug

    do_flip = np.random.uniform(0.0, 1.0) < 0.5
    h, w = rgb.shape[0], rgb.shape[1]
    
    transforms_list = [
        transforms.BottomCrop((cfg.NK.KITTI.bottom_height, cfg.NK.KITTI.bottom_width)),
        transforms.HorizontalFlip(do_flip)]

    transform_geometric = transforms.Compose(transforms_list)

    # random crop
    b_h = cfg.NK.KITTI.bottom_height
    b_w = cfg.NK.KITTI.bottom_width
    c_h = cfg.NK.KITTI.crop_height
    c_w = cfg.NK.KITTI.crop_width
    i = np.random.randint(0, b_h - c_h + 1)
    j = np.random.randint(0, b_w - c_w + 1)

    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - setting.color_jitter),
                                       1 + setting.color_jitter)
        contrast = np.random.uniform(max(0, 1 - setting.color_jitter), 1 + setting.color_jitter)
        saturation = np.random.uniform(max(0, 1 - setting.color_jitter),
                                       1 + setting.color_jitter)
        transform_rgb = transforms.Compose([
            transforms.ColorJitterV2(brightness, contrast, saturation, 0),
            transform_geometric])
        rgb = transform_rgb(rgb)

        if rgb.ndim == 3:
            rgb = rgb[i:i + c_h, j:j + c_w, :]
        elif rgb.ndim == 2:
            rgb = rgb[i:i + c_h, j:j + c_w]

    if sparse is not None:
        sparse = transform_geometric(sparse)
        if sparse.ndim == 3:
            sparse = sparse[i:i + c_h, j:j + c_w, :]
        elif sparse.ndim == 2:
            sparse = sparse[i:i + c_h, j:j + c_w]

    if target is not None:
        target = transform_geometric(target)
        if target.ndim == 3:
            target = target[i:i + c_h, j:j + c_w, :]
        elif target.ndim == 2:
            target = target[i:i + c_h, j:j + c_w]
    
    if K is not None:
        y_start = (h - cfg.NK.KITTI.bottom_height) # BottomCrop!!!
        x_start = (w - cfg.NK.KITTI.bottom_width) // 2
        y2 = i
        x2 = j
        K = K + [[0.0, 0.0, -x_start-x2],
                 [0.0, 0.0, -y_start-y2],
                 [0.0, 0.0, 0.0]]

    return rgb, sparse, target, K

class NKDataset(data.Dataset):
    def __init__(self, cfg, mode='train'):
        super(NKDataset, self).__init__()

        self.cfg = cfg
        self.mode = mode
        self.ncfg = cfg.NK.NYUv2
        self.kcfg = cfg.NK.KITTI
        self.rate = cfg.NK.rate

        self.setting = cfg.train.dataset.img_aug
        self.jitter = self.setting.color_jitter
        self.degree = self.setting.degree
        self.scale = self.setting.scale

        self.data_files, self.rgb_files, self.sparse_files, self.target_files \
            = self.load_path(cfg.NK.trainlist)
        self.transform = kitti_train_transform

        self.crop_size = (cfg.NK.crop_height, cfg.NK.crop_width)

        self.NYU_h = cfg.NK.crop_height
        self.NYU_w = cfg.NK.crop_width
        self.KITTI_rh = cfg.NK.KITTI.resize_height
        self.KITTI_rw = cfg.NK.KITTI.resize_width
        self.KITTI_ch = cfg.NK.KITTI.crop_height
        self.KITTI_cw = cfg.NK.KITTI.crop_width
        self.KITTI_sh = cfg.NK.KITTI.resize_height / cfg.NK.KITTI.crop_height
        self.KITTI_sw = cfg.NK.KITTI.resize_width / cfg.NK.KITTI.crop_width

        if cfg.MDEBranch.backbone == 'midas':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.NormalizeTensor(cfg.MDEBranch.midas.mean, cfg.MDEBranch.midas.std),
                transforms.ConvertFloat()])
            self.kitti_mde = transforms.Compose([
                transforms.ResizeV2((cfg.MDEBranch.midas.resize_h, cfg.MDEBranch.midas.resize_w), 1)])
            self.nyu_mde = transforms.Compose([
                transforms.ResizeV1((cfg.MDEBranch.midas.resize_h, cfg.MDEBranch.midas.resize_w), Image.BILINEAR)])
        
        elif cfg.MDEBranch.backbone == 'depthanything' or cfg.MDEBranch.backbone == 'depthanythingv2':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.NormalizeTensor(cfg.MDEBranch.depany.mean, cfg.MDEBranch.depany.std),
                transforms.ConvertFloat()])

            self.kitti_mde = transforms.Compose([
                transforms.ResizeV4(
                width=cfg.MDEBranch.depany.resize_w, 
                height=cfg.MDEBranch.depany.resize_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=3,)])
            
            self.nyu_mde = transforms.Compose([
                transforms.ResizeV3(
                width=cfg.MDEBranch.depany.resize_w, 
                height=cfg.MDEBranch.depany.resize_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=Image.BICUBIC,)])
        
        elif cfg.MDEBranch.backbone == 'depthpro':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.NormalizeTensor(cfg.MDEBranch.depthpro.mean, cfg.MDEBranch.depthpro.std),
                transforms.ConvertFloat()])

            self.kitti_mde = transforms.Compose([
                transforms.ResizeV2((cfg.MDEBranch.depthpro.resize_h, cfg.MDEBranch.depthpro.resize_w), 1)])
            self.nyu_mde = transforms.Compose([
                transforms.ResizeV1((cfg.MDEBranch.depthpro.resize_h, cfg.MDEBranch.depthpro.resize_w), Image.BILINEAR)])
        
        elif cfg.MDEBranch.backbone == 'metricv2':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.NormalizeTensor(cfg.MDEBranch.metricv2.mean, cfg.MDEBranch.metricv2.std),
                transforms.ConvertFloat()])

            self.kitti_mde = transforms.Compose([
                transforms.ResizeV2((cfg.MDEBranch.metricv2.resize_h, cfg.MDEBranch.metricv2.resize_w), 1)])
            self.nyu_mde = transforms.Compose([
                transforms.ResizeV1((cfg.MDEBranch.metricv2.resize_h, cfg.MDEBranch.metricv2.resize_w), Image.BILINEAR)])
            
            self.data_basic = cfg.NK.data_basic
        
        elif cfg.MDEBranch.backbone == 'promptda':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.ConvertFloat()])

            self.kitti_mde = transforms.Compose([
                transforms.ResizeV4(
                width=cfg.MDEBranch.promptda.resize_w, 
                height=cfg.MDEBranch.promptda.resize_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=3,)])
            
            self.nyu_mde = transforms.Compose([
                transforms.ResizeV3(
                width=cfg.MDEBranch.promptda.resize_w, 
                height=cfg.MDEBranch.promptda.resize_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=Image.BICUBIC,)])
        
        elif cfg.MDEBranch.backbone == 'unidepth':
            self.trans_rgb2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.ConvertFloat()])

            self.kitti_mde = transforms.Compose([
                transforms.ResizeV4(
                width=cfg.MDEBranch.unidepth.resize_w, 
                height=cfg.MDEBranch.unidepth.resize_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=3,)])
            
            self.nyu_mde = transforms.Compose([
                transforms.ResizeV3(
                width=cfg.MDEBranch.unidepth.resize_w, 
                height=cfg.MDEBranch.unidepth.resize_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=Image.BICUBIC,)])

        self.backbone = cfg.MDEBranch.backbone
        
        self.trans_tensor = transforms.ToTensor()
        self.trans_resize = transforms.Compose([
                transforms.ResizeV2((self.KITTI_rh, self.KITTI_rw), 1)])

    def load_path(self, list_filename):
        assert os.path.exists(list_filename), "file not found: {}".format(list_filename)
        with open(list_filename, "r") as f:
            lines = [line.rstrip() for line in f.readlines()]

        splits = [line.split() for line in lines]
        data_files = [x[0] for x in splits]
        rgb_files = [x[1] for x in splits]
        sparse_files = [x[2] for x in splits]
        target_files = [x[3] for x in splits]
        return data_files, rgb_files, sparse_files, target_files 

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, index):
        dataset = self.data_files[index]
        
        if dataset == 'NYUv2':
            inputs = self.get_nyu(index)
            inputs['dataset'] = torch.zeros(1)
            
        elif dataset == 'KITTI':
            inputs = self.get_kitti(index)
            inputs['dataset'] = torch.ones(1)
        return inputs

    def get_nyu(self, index):
        inputs = {}
        path_file = os.path.join(self.ncfg.path, self.rgb_files[index])
        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
        dep_h5 = f['depth'][:]

        rgb = Image.fromarray(rgb_h5, mode='RGB')
        target = Image.fromarray(dep_h5.astype('float32'), mode='F')
        K = np.reshape(K_nyu, [3, 3])
        height = self.ncfg.height

        cur_scale = np.random.uniform(1.0, self.scale)
        if index == 24132:
            cur_scale = 1
        height_scale = int(height * cur_scale)

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

        K[0][0] = K[0][0] * cur_scale
        K[1][1] = K[1][1] * cur_scale
        if self.backbone == 'metricv2':
            intrinsic = [K[0][0], K[1][1], K[0][2], K[1][2]]
            rgb_mde = self.nyu_mde(rgb)
            rgb_mde = np.array(rgb_mde)
            rgb_mde, cam_models_stacks, pad, label_scale_factor = transform_data_scalecano(rgb_mde, intrinsic, self.data_basic)     
            inputs['cam_model'] = cam_models_stacks
            inputs['pad_info'] = pad
            inputs['scale_info'] = label_scale_factor
            inputs['normalize_scale'] = self.data_basic.depth_range[1]
        else:
            rgb_mde = self.trans_rgb2tensor(self.nyu_mde(rgb))
        rgb = self.trans_rgb2tensor(rgb)

        target = trans_dep(target)
        target = target / cur_scale

        inputs['rgb'] = rgb
        inputs['rgb_mde'] = rgb_mde
        inputs['target'] = target

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
                num = int(self.rate*self.NYU_h*self.NYU_w)
            sparse, _ = self.get_sparse_depth(inputs['target'].clone(), num)
            mask = sparse > 0.000001

        sparse = target.clone()
        do_noiseblur = np.random.uniform(0.0, 1.0) < 0.10
        if do_noiseblur:
            sparse_noise = self._noiseblur(sparse)
        else:
            sparse_noise = sparse
        inputs['sparse'] = sparse_noise * mask

        inputs['K'] = torch.from_numpy(K)

        sparse_ip2 = ip(inputs['sparse'].clone().numpy().squeeze(0), max_depth=inputs['sparse'].max())
        inputs['ip'] = self.trans_tensor(sparse_ip2)

        return inputs

    def get_kitti(self, index):
        inputs = {}
        rgb_file = os.path.join(self.kcfg.raw, self.rgb_files[index])
        sparse_file = os.path.join(self.kcfg.path, self.sparse_files[index])
        target_file = os.path.join(self.kcfg.path, self.target_files[index])
        
        rgb = self.get_rgb(rgb_file)
        sparse = self.get_depth(sparse_file) 
        target = self.get_depth(target_file)
        K = np.reshape(Ks[self.rgb_files[index].split('/')[0]], [3, 3])

        rgb, sparse, target, K = self.transform(rgb, sparse, target, K, self.cfg)
        
        rgb = self.trans_resize(rgb)
        target = self.trans_tensor(target)
        target = F.interpolate(target.unsqueeze(0), size=(self.KITTI_rh, self.KITTI_rw), mode="nearest").squeeze(0) 
        sparse = self.trans_tensor(sparse)
        sparse = F.interpolate(sparse.unsqueeze(0), size=(self.KITTI_rh, self.KITTI_rw), mode="nearest").squeeze(0)

        do_sparse = np.random.uniform(0.0, 1.0)
        if do_sparse < 0.1:
            mask = target > 0.00001
            num = int(mask.sum() / 4)
            sparse_rate, _ = self.get_sparse_depth(target.clone(), num)
            inputs['sparse'] = sparse_rate
        elif do_sparse < 0.7:
            do_noise = np.random.uniform(0.0, 1.0) < 0.50
            if do_noise:
                sparse = self._noise(sparse)
            else:
                sparse = target

            do_aug = np.random.uniform(0.0, 1.0) < 0.10
            if do_aug:
                num = 400
            else:
                num = int(self.rate*self.KITTI_rh*self.KITTI_rw)
            sparse_rate, _ = self.get_sparse_depth(sparse.clone(), num)
            inputs['sparse'] = sparse_rate
        else:
            inputs['sparse'] = sparse
        
        K[0][0] = K[0][0] * self.KITTI_sw
        K[0][2] = K[0][2] * self.KITTI_sw
        K[1][1] = K[1][1] * self.KITTI_sh
        K[1][2] = K[1][2] * self.KITTI_sh

        if self.backbone == 'metricv2':
            intrinsic = [K[0][0], K[1][1], K[0][2], K[1][2]]
            rgb_mde = self.kitti_mde(rgb)
            rgb_mde = np.array(rgb_mde) * 255.
            rgb_mde, cam_models_stacks, pad, label_scale_factor = transform_data_scalecano(rgb_mde, intrinsic, self.data_basic)     
            inputs['cam_model'] = cam_models_stacks
            inputs['pad_info'] = pad
            inputs['scale_info'] = label_scale_factor
            inputs['normalize_scale'] = self.data_basic.depth_range[1]
        else:
            rgb_mde = self.trans_rgb2tensor(self.kitti_mde(rgb))
        rgb = self.trans_rgb2tensor(rgb)

        inputs['rgb'] = rgb
        inputs['rgb_mde'] = rgb_mde
        inputs['target'] = target          
        
        
        inputs['K'] = self.trans_tensor(K).squeeze(0)

        sparse_ip = ip(inputs['sparse'].clone().numpy().squeeze(0), max_depth=inputs['sparse'].max())
        inputs['ip'] = self.trans_tensor(sparse_ip)
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
    
    def _noise(self, raw, p_noise=0.5):
        raw_shape = raw.shape
        raw_min = raw.min()
        raw_max = raw.max()
        # add noise
        if np.random.uniform(0.0, 1.0) < p_noise:
            gaussian_noise = torch.ones(raw_shape).normal_(0, np.random.uniform(0.01, 0.1))
            gaussian_noise = self._sample(gaussian_noise, np.random.uniform(0.25, 0.75))
            raw = raw + gaussian_noise
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
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255, \
            "np.max(depth_png)={}, path={}".format(np.max(depth_png), file)

        depth = depth_png.astype(np.float32) / 256.
        # depth[depth_png == 0] = -1.
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
