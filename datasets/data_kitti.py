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

def train_transform(rgb, sparse, target, K, cfg):
    setting = cfg.train.dataset.img_aug

    do_flip = np.random.uniform(0.0, 1.0) < 0.5
    h, w = rgb.shape[0], rgb.shape[1]
    
    transforms_list = [
        transforms.BottomCrop((cfg.KITTI.bottom_height, cfg.KITTI.bottom_width)),
        transforms.HorizontalFlip(do_flip)]

    transform_geometric = transforms.Compose(transforms_list)

    # random crop
    b_h = cfg.KITTI.bottom_height
    b_w = cfg.KITTI.bottom_width
    c_h = cfg.KITTI.crop_height
    c_w = cfg.KITTI.crop_width
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
        y_start = (h - cfg.KITTI.bottom_height) # BottomCrop!!!
        x_start = (w - cfg.KITTI.bottom_width) // 2
        y2 = i
        x2 = j
        K = K + [[0.0, 0.0, -x_start-x2],
                 [0.0, 0.0, -y_start-y2],
                 [0.0, 0.0, 0.0]]

    return rgb, sparse, target, K

def val_transform(rgb, sparse, target, K, cfg):
    h, w = rgb.shape[0], rgb.shape[1]
    
    # transforms_list = [
    #     transforms.BottomCrop((cfg.KITTI.bottom_height, cfg.KITTI.bottom_width))]

    # transform_geometric = transforms.Compose(transforms_list)

    # rgb = transform_geometric(rgb)
    # sparse = transform_geometric(sparse)
    # target = transform_geometric(target)

    y_start = (h - cfg.KITTI.bottom_height) # BottomCrop!!!
    x_start = (w - cfg.KITTI.bottom_width) // 2
    K = K + [[0.0, 0.0, -x_start],
             [0.0, 0.0, -y_start],
             [0.0, 0.0, 0.0]]
            
    return rgb, sparse, target, K

class KittiDataset(data.Dataset):
    def __init__(self, cfg, mode='train'):
        super(KittiDataset, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.path = cfg.KITTI.path
        self.raw = cfg.KITTI.raw
        self.rate = cfg.KITTI.rate

        self.rh = cfg.KITTI.resize_height
        self.rw = cfg.KITTI.resize_width
        self.ch = cfg.KITTI.crop_height
        self.cw = cfg.KITTI.crop_width
        self.sh = cfg.KITTI.resize_height / cfg.KITTI.crop_height
        self.sw = cfg.KITTI.resize_width / cfg.KITTI.crop_width

        self.valrgb = cfg.KITTI.valrgb
        self.valgt = cfg.KITTI.valgt
        self.valraw = cfg.KITTI.valraw
        self.valk = cfg.KITTI.valk

        self.setting = cfg.train.dataset.img_aug
        self.jitter = self.setting.color_jitter

        if mode== 'train':
            self.rgb_files, self.sparse_files, self.target_files \
                = self.load_path(cfg.KITTI.trainlist)
            self.transform = train_transform
        elif mode== 'val':
            self.rgb_files, self.sparse_files, self.target_files, self.intrinsic_files \
                = self.load_path(cfg.KITTI.vallist)
            self.transform = val_transform
        elif mode== 'test':
            pass

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
            
            self.data_basic = cfg.KITTI.data_basic
        
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

        self.trans_resize = transforms.Compose([
                transforms.ResizeV2((self.rh, self.rw), 1)])
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
        if self.mode == 'train':
            target_files = [x[2] for x in splits]
            return rgb_files, sparse_files, target_files
        elif self.mode == 'val':
            target_files = [x[2] for x in splits]
            intrinsic_files = [x[3] for x in splits]
            return rgb_files, sparse_files, target_files, intrinsic_files
        elif self.mode == 'test':
            intrinsic_files = [x[2] for x in splits]
            return rgb_files, sparse_files, intrinsic_files

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, index):
        inputs={}
        if self.mode == 'train':
            rgb_file = os.path.join(self.raw, self.rgb_files[index])
            sparse_file = os.path.join(self.path, self.sparse_files[index])
            target_file = os.path.join(self.path, self.target_files[index])
            
            rgb = self.get_rgb(rgb_file)
            sparse = self.get_depth(sparse_file) 
            target = self.get_depth(target_file)
            K = np.reshape(Ks[self.rgb_files[index].split('/')[0]], [3, 3])

            rgb, sparse, target, K = self.transform(rgb, sparse, target, K, self.cfg)
            
            rgb = self.trans_resize(rgb)
            target = self.trans_tensor(target)
            target = F.interpolate(target.unsqueeze(0), size=(self.rh, self.rw), mode="nearest").squeeze(0) 

            do_sparse = np.random.uniform(0.0, 1.0)
            if do_sparse < 0.1:
                sparse = target
                raw_depth = target.squeeze()
                masks = []
                pseudo_sample = {'rgb': rgb ,'raw_depth': raw_depth}
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
                pseudo_depth = self.trans_tensor(pseudo_depth)
                mask = pseudo_depth > 0.000001

                do_noiseblur = np.random.uniform(0.0, 1.0) < 0.20
                if do_noiseblur:
                    sparse_noise = self._noiseblur(sparse)
                else:
                    sparse_noise = sparse
                inputs['sparse'] = sparse_noise*mask

            elif do_sparse < 0.7:
                sparse = target
                do_noiseblur = np.random.uniform(0.0, 1.0) < 0.20
                if do_noiseblur:
                    sparse_noise = self._noiseblur(sparse)
                else:
                    sparse_noise = sparse

                num = int(self.rate*self.rh*self.rw)
                sparse_rate, _ = self.get_sparse_depth(sparse_noise.clone(), num)
                inputs['sparse'] = sparse_rate
            else:
                sparse = self.trans_tensor(sparse)
                sparse = F.interpolate(sparse.unsqueeze(0), size=(self.rh, self.rw), mode="nearest").squeeze(0)
                inputs['sparse'] = sparse
            
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

            inputs['rgb'] = rgb
            inputs['rgb_mde'] = rgb_mde
            inputs['target'] = target          
            
            K[0][0] = K[0][0] * self.sw
            K[0][2] = K[0][2] * self.sw
            K[1][1] = K[1][1] * self.sh
            K[1][2] = K[1][2] * self.sh
            inputs['K'] = self.trans_tensor(K).squeeze(0)

            sparse_ip = ip(inputs['sparse'].clone().numpy().squeeze(0), max_depth=inputs['sparse'].max())
            sparse_ip = self.trans_tensor(sparse_ip)
            inputs['ip'] = F.interpolate(sparse_ip.unsqueeze(0), size=(self.rh, self.rw), mode="nearest").squeeze(0)

            # import matplotlib.pyplot as plt
            # import matplotlib 
            # from matplotlib.backends.backend_agg import FigureCanvasAgg
            # matplotlib.use('pdf')

            # plt.imshow(inputs['sparse'].cpu().squeeze(), cmap="jet") 
            # plt.axis('off')
            # plt.savefig('./save/'+n+'_{:0>4d}.png'.format(index), bbox_inches = 'tight', pad_inches = 0, dpi=32)
            
        elif self.mode == 'val':
            rgb_file = os.path.join(self.valrgb, self.rgb_files[index])
            sparse_file = os.path.join(self.valraw, self.sparse_files[index])
            target_file = os.path.join(self.valgt, self.target_files[index])
            intrinsic_file = os.path.join(self.valk, self.intrinsic_files[index])

            rgb = self.get_rgb(rgb_file)
            sparse = self.get_depth(sparse_file) 
            target = self.get_depth(target_file)
            K = np.reshape(np.loadtxt(intrinsic_file, dtype=np.float32), [3, 3])
            
            rgb, sparse, target, K = self.transform(rgb, sparse, target, K, self.cfg)
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
            sample_factor = 2 ** (np.random.randint(1, 3))
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
    