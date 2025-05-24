import os 
import time
import json
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from torch import distributed as dist
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import __datasets__
from network.ipde_5c_l1_rezero_un_3 import DCNet
from util import *
from tool import *
import logging
import csv
from PIL import Image

class Trainer:
    def __init__(self, cfg):
        self.logger = self.get_logger()
        self.logger.info('Get logger')

        self.logger.info('Set seed')
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        if cfg.threads is not None:
            torch.set_num_threads(cfg.threads)
        else:
            torch.set_num_threads(cfg.train.batch_size)

        self.cfg = cfg      
        
        # model
        self.iter_step = 0
        self.start_epoch = 0
        self.total_epoch = cfg.train.total_epoch
        self.epoch = 0

        self.logger.info('Create network')
        self.net = DCNet(cfg)

        self.logger.info('Create optimizer')
        if cfg.train.optimizer.name == 'adamw':
            # for p in self.net.parameters():
            #     p.requires_grad = False
            # for p in self.net.total_offset.parameters():
            #     p.requires_grad = True
            # model_named_params = [
            #     p for _, p in self.net.named_parameters() if p.requires_grad ]

            self.optimizer = optim.AdamW(
                self.net.parameters(), 
                eps=cfg.train.optimizer.eps,
                betas=cfg.train.optimizer.betas,
                lr=cfg.train.base_lr,
                weight_decay=cfg.train.optimizer.weight_decay)
        elif cfg.train.optimizer.name == 'adam':
            self.optimizer = optim.Adam(
                self.net.parameters(), 
                cfg.train.base_lr, 
                betas=cfg.train.optimizer.betas)
        else:
            raise ValueError(f'Unsupported optimizer: {cfg.train.optimizer.name}')
        
        if cfg.train.lr_scheduler.name == 'cosine':
            self.lr_scheduler = WarmupCosineLR(self.optimizer, cfg.train.min_lr, cfg.train.max_lr, \
                     warm_up=cfg.train.warmup_epochs, T_max=cfg.train.total_epoch, start_ratio=cfg.train.start_ratio)
        elif cfg.train.lr_scheduler.name == 'stage':
            self.lr_epoch = cfg.train.lr_scheduler.lr_epoch
            self.lrs = cfg.train.lr_scheduler.lrs
            assert len(self.lr_epoch) == len(self.lrs)
            self.learning_rate = self.lrs[0]
        else:
            raise ValueError(f'Unsupported schedule: {cfg.train.lr_scheduler.name}')

        self.net = self.net.cuda()
        if cfg.load_ckpt is not None:
            self.load_model()
            self.logger.info('Load model completed')

        self.gpu_num = torch.cuda.device_count()
        if self.gpu_num > 1:
            self.logger.info('Device number {}'.format(self.gpu_num))
            self.net = nn.DataParallel(self.net)
        self.logger.info('To device')

        self.print_params()
        # self.print_flops()

        # train data
        dataset = __datasets__[cfg.train.dataset.name]
        train_dataset = dataset(cfg, mode='train')
        self.train_loader = DataLoader(train_dataset, cfg.train.batch_size, \
                            num_workers=cfg.train.batch_size, \
                            shuffle=True, drop_last=True)    
        self.logger.info("Train dataset {:d} and {:d}".format(len(train_dataset), len(self.train_loader)))

        # log
        now = datetime.now().strftime("%Y-%m-%d")
        self.log_path = os.path.join(cfg.log.dir, now, cfg.model_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        self.logger.info("Log path: {}".format(self.log_path))

        # nyu data
        dataset = __datasets__[cfg.NYUv2.name]
        nyu_dataset = dataset(cfg, mode='val')
        self.nyu_loader = DataLoader(nyu_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_nyu = os.path.join(cfg.log.dir, now, 'nyu-result.csv')
        with open(self.csvfile_nyu, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load nyu {}".format(len(self.nyu_loader)))

        # diode_in
        dataset = __datasets__[cfg.DIODEi.name]
        diodei_dataset = dataset(cfg, mode='val')
        self.diodei_loader = DataLoader(diodei_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_diodei = os.path.join(cfg.log.dir, now, 'diodei-result.csv')
        with open(self.csvfile_diodei, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load diode indoor {}".format(len(self.diodei_loader)))

        # sunrgbd
        dataset = __datasets__[cfg.SUNRGBD.name]
        surgbd_dataset = dataset(cfg, mode='val')
        self.sunrgbd_loader = DataLoader(surgbd_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_sunrgbd = os.path.join(cfg.log.dir, now, 'sunrgbd-result.csv')
        with open(self.csvfile_sunrgbd, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load SUNRGBD {}".format(len(self.sunrgbd_loader)))

        # scannet
        dataset = __datasets__[cfg.SCANNET.name]
        scannet_dataset = dataset(cfg, mode='val')
        self.scannet_loader = DataLoader(scannet_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_scannet = os.path.join(cfg.log.dir, now, 'scannet-result.csv')
        with open(self.csvfile_scannet, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load SCANNET {}".format(len(self.scannet_loader)))

        # middlebury
        dataset = __datasets__[cfg.MIDDLEBURY.name]
        middlebury_dataset = dataset(cfg, mode='val')
        self.middlebury_loader = DataLoader(middlebury_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_middlebury = os.path.join(cfg.log.dir, now, 'middlebury-result.csv')
        with open(self.csvfile_middlebury, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load MIDDLEBURY {}".format(len(self.middlebury_loader)))

        # hypersim
        dataset = __datasets__[cfg.HYPERSIM.name]
        hypersim_dataset = dataset(cfg, mode='val')
        self.hypersim_loader = DataLoader(hypersim_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_hypersim = os.path.join(cfg.log.dir, now, 'hypersim-result.csv')
        with open(self.csvfile_hypersim, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load HYPERSIM {}".format(len(self.hypersim_loader)))

        # eth3d
        eth3d_dataset = __datasets__[cfg.ETH3D.name](cfg, mode='val')
        self.eth3d_loader = DataLoader(eth3d_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_eth3d = os.path.join(cfg.log.dir, now, 'eth3d-result.csv')
        with open(self.csvfile_eth3d, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load eth3d {}".format(len(self.eth3d_loader)))

        # void 
        void_dataset = __datasets__[cfg.VOID1500.name](cfg, mode='val')
        self.void_loader = DataLoader(void_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_void = os.path.join(cfg.log.dir, now, 'void-result.csv')
        with open(self.csvfile_void, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load void {}".format(len(self.void_loader)))

        # kitti
        kitti_dataset = __datasets__[cfg.KITTI.name](cfg, mode='val')
        self.kitti_loader = DataLoader(kitti_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_kitti = os.path.join(cfg.log.dir, now, 'kitti-result.csv')
        with open(self.csvfile_kitti, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load kitti {}".format(len(self.kitti_loader)))

        # ds
        ds_dataset = __datasets__[cfg.DrivingStereo.name](cfg, mode='val')
        self.ds_loader = DataLoader(ds_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_ds = os.path.join(cfg.log.dir, now, 'ds-result.csv')
        with open(self.csvfile_ds, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load drivingstereo {}".format(len(self.ds_loader)))

        # diode out
        dataset = __datasets__[cfg.DIODEo.name]
        diodeo_dataset = dataset(cfg, mode='val')
        self.diodeo_loader = DataLoader(diodeo_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_diodeo = os.path.join(cfg.log.dir, now, 'diodeo-result.csv')
        with open(self.csvfile_diodeo, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load diode outdoor {}".format(len(self.diodeo_loader)))

        # argoverse
        dataset = __datasets__[cfg.ARGOVERSE.name]
        arg_dataset = dataset(cfg, mode='val')
        self.arg_loader = DataLoader(arg_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_arg = os.path.join(cfg.log.dir, now, 'arg-result.csv')
        with open(self.csvfile_arg, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load ARGOVERSE outdoor {}".format(len(self.arg_loader)))
        
        # vkitti2
        dataset = __datasets__[cfg.VKITTI2.name]
        vkitti2_dataset = dataset(cfg, mode='val')
        self.vkitti2_loader = DataLoader(vkitti2_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_vkitti2 = os.path.join(cfg.log.dir, now, 'vkitti2-result.csv')
        with open(self.csvfile_vkitti2, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load VKITTI2 {}".format(len(self.vkitti2_loader)))

        # dimli 
        dimli_dataset = __datasets__[cfg.DIMLi.name](cfg, mode='val')
        self.dimli_loader = DataLoader(dimli_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_dimli = os.path.join(cfg.log.dir, now, 'dimli-result.csv')
        with open(self.csvfile_dimli, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load dimli {}".format(len(self.dimli_loader)))

        # cityscape 
        cityscape_dataset = __datasets__[cfg.Cityscape.name](cfg, mode='val')
        self.cityscape_loader = DataLoader(cityscape_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_cityscape = os.path.join(cfg.log.dir, now, 'cityscape-result.csv')
        with open(self.csvfile_cityscape, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load Cityscape {}".format(len(self.cityscape_loader)))

        # TOFDC 
        tofdc_dataset = __datasets__[cfg.TOFDC.name](cfg, mode='val')
        self.tofdc_loader = DataLoader(tofdc_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_tofdc = os.path.join(cfg.log.dir, now, 'tofdc-result.csv')
        with open(self.csvfile_tofdc, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load TOFDC {}".format(len(self.tofdc_loader)))

        # HAMMER
        hammer_dataset = __datasets__[cfg.HAMMER.name](cfg, mode='val')
        self.hammer_loader = DataLoader(hammer_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_hammer = os.path.join(cfg.log.dir, now, 'hammer-result.csv')
        with open(self.csvfile_hammer, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load HAMMER {}".format(len(self.hammer_loader)))

        # Stanford
        stanford_dataset = __datasets__[cfg.Stanford.name](cfg, mode='val')
        self.stanford_loader = DataLoader(stanford_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_stanford = os.path.join(cfg.log.dir, now, 'stanford-result.csv')
        with open(self.csvfile_stanford, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load Stanford {}".format(len(self.stanford_loader)))

        # KITTI360
        kitti360_dataset = __datasets__[cfg.KITTI360.name](cfg, mode='val')
        self.kitti360_loader = DataLoader(kitti360_dataset, 1, num_workers=1, shuffle=False, drop_last=False)
        self.csvfile_kitti360 = os.path.join(cfg.log.dir, now, 'kitti360-result.csv')
        with open(self.csvfile_kitti360, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "{:6}".format('epoch'),
                    "{:10}".format('rmse'), 
                    "{:10}".format('mae'), 
                    "{:10}".format('rel'),  
                    "{:10}".format('irmse'), 
                    "{:10}".format('imae'),
                    "{:10}".format('d1'), 
                    "{:10}".format('d2'), 
                    "{:10}".format('d3')])
                writer.writeheader()
        self.logger.info("load KITTI360 {}".format(len(self.kitti360_loader)))

        # best
        self.best_results = {
            'epoch': -1,
            'mae': np.infty, 'rmse': np.infty, 'imae': np.infty, 'irmse': np.infty,
            'rel': np.infty, 'd1': -1, 'd2': -1, 'd3': -1} 

    def get_logger(self):
        logger_name = "main-logger"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        return logger
    
    def save_csv(self, results, csvfile_name):
        fieldnames = ['epoch', 'rmse', 'mae', 'rel', 'irmse', 'imae', 'd1', 'd2', 'd3']
        with open(csvfile_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for i, (a, b, c, d, e, f, g, h) in enumerate(zip(results['mae'], results['rmse'], results['imae'], results['irmse'],
                results['rel'], results['d1'], results['d2'], results['d3'])):
                writer.writerow({
                    'epoch': "%-6d" % self.epoch,
                    'rmse': "%-10.5f" % round(b, 5),
                    'mae': "%-10.5f" % round(a, 5),
                    'rel': "%-10.5f" % round(e, 5),
                    'irmse': "%-10.5f" % round(d, 5),
                    'imae': "%-10.5f" % round(c, 5),
                    'd1': "%-10.5f" % round(f, 5),
                    'd2': "%-10.5f" % round(g, 5),
                    'd3': "%-10.5f" % round(h, 5),})
    
    def print_params(self):
        parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        parameters =np.sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        self.logger.info('\t=> Parameters: %.4fM' % parameters)

    def print_flops(self):
        list_conv = []
        def conv_hook(self, input, output):
            # Can have multiple inputs, getting the first one
            input = input[0]

            batch_size = input.shape[0]
            output_dims = list(output.shape[2:])

            kernel_dims = list(self.kernel_size)
            in_channels = self.in_channels
            out_channels = self.out_channels
            groups = self.groups

            filters_per_channel = out_channels // groups
            conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels * filters_per_channel

            active_elements_count = batch_size * int(np.prod(output_dims))

            overall_conv_flops = conv_per_position_flops * active_elements_count

            bias_flops = 0

            if self.bias is not None:

                bias_flops = out_channels * active_elements_count

            overall_flops = overall_conv_flops + bias_flops
            

            list_conv.append(int(overall_flops))

        list_linear = []
        def linear_hook(self, input, output):
            input = input[0]
            # pytorch checks dimensions, so here we don't care much
            output_last_dim = output.shape[-1]
            bias_flops = output_last_dim if self.bias is not None else 0
            
            list_linear.append(int(np.prod(input.shape) * output_last_dim + bias_flops))
        
        list_pool = []
        def pool_hook(self, input, output):
            input = input[0]
            list_pool.append(int(np.prod(input.shape)))

        def foo(net): 
            childrens = list(net.children())
            if not childrens:
                if isinstance(net, torch.nn.Conv1d):
                    net.register_forward_hook(conv_hook)
                if isinstance(net, torch.nn.Conv2d):
                    net.register_forward_hook(conv_hook)
                if isinstance(net, torch.nn.ConvTranspose2d):
                    net.register_forward_hook(conv_hook)
                if isinstance(net, torch.nn.Linear):
                    net.register_forward_hook(linear_hook)
                if isinstance(net, torch.nn.Linear):
                    net.register_forward_hook(pool_hook)
                return
            for c in childrens:
                foo(c)

        foo(self.net)

        self.net.eval()
        inputs = {}
        inputs['rgb'] = torch.ones(1, 3, 1216, 352).cuda()
        inputs['sparse'] = torch.ones(1, 1, 1216, 352).cuda()
        inputs['K'] = torch.rand(1, 1, 3, 3).cuda()
        out = self.net(inputs, mode='test')
        total_flops = sum(sum(i) for i in [list_conv, list_linear, list_pool])

        self.logger.info('\t=> {}*{} image FLOPs is {}G'.format(352, 1216, total_flops/1_000_000_000))
   
    def warmup(self, data_len):
        all_steps = data_len * self.cfg.warmup_epoch
        self.warmup_step = self.warmup_step + 1
        self.learning_rate = (self.cfg.warmup_lr[1]-self.cfg.warmup_lr[0]) / all_steps * self.warmup_step+self.cfg.warmup_lr[0]
        # Update optimizer learning rates
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate
        self.logger.info('\t=> Warmup step: {}, learning rate: {}'.format(self.warmup_step, self.learning_rate))
    
    def adjust_lr(self):
        if self.cfg.train.lr_scheduler.name == 'stage':
            for i in range(len(self.lr_epoch)):
                if self.epoch >= self.lr_epoch[i]:
                    self.learning_rate = self.lrs[i]
            for g in self.optimizer.param_groups:
                g['lr'] = self.learning_rate

    def set_train(self):
        self.net.train()

    def set_eval(self):
        self.net.eval()

    def run(self):
        self.start_time = time.time()
        if self.cfg.run_mode=='train':
            self.logger.info("Train model!")
            if self.start_epoch > 0:
                self.logger.info("Start epoch is not 0, adjust learning rate!")

            for self.epoch in range(self.start_epoch, self.total_epoch):
                if self.cfg.train.lr_scheduler.name == 'cosine':
                    self.lr_scheduler.step()
                elif self.cfg.train.lr_scheduler.name == 'stage':
                    self.adjust_lr()

                self.logger.info("Training epoch: {}".format(self.epoch))
                self.logger.info("learning rate: {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.train()
                torch.cuda.empty_cache()

                self.save_model()

                self.val_nyu()
                self.val_kitti()
                # self.val_diodei()
                # self.val_sunrgbd()
                # self.val_scannet()
                # self.val_middlebury()
                # self.val_hypersim()
                # self.val_eth3d()
                self.val_void()
                # self.val_ds()
                # self.val_diodeo()
                # self.val_argoverse()
                # self.val_vkitti2()
                # self.val_dimli()
                # self.val_cityscape()
                # self.val_tofdc()
                # self.val_hammer()

                torch.cuda.empty_cache()
                
        elif self.cfg.run_mode=='val':
            self.logger.info("Val model!")
            # self.val_nyu()
            # self.val_kitti()
            # self.val_void()
            # self.val_sunrgbd()
            # self.val_dimli()
            # self.val_tofdc()
            # self.val_hammer()
            # self.val_ds()
            
            # self.val_diodeo()
            # self.val_diodei()
            # self.val_scannet()
            # self.val_middlebury()
            # self.val_eth3d()
            # self.val_vkitti2()
            # self.val_hypersim()
            # self.val_cityscape()
            # self.val_argoverse()  # metric3d CUDA out of memory in 189
            self.val_stanford()
            # self.val_kitti360()

        elif self.cfg.run_mode=='test':
            self.logger.info("Test model!")
            self.test()
        else:
            print('!!!!!!!!!!')
            raise ValueError

    def train(self):
        self.set_train()

        if self.gpu_num > 1:
            self.net.module.mde.apply(fix_bn)
            self.logger.info('\t=> GPU NUM {}, Fix BN'.format(self.gpu_num))
        else:
            self.net.mde.apply(fix_bn)
            self.logger.info('\t=> GPU NUM 1, Fix BN')
        # self.net.segmant.apply(fix_bn)
        # self.net.total_offset.apply(release_bn)

        data_loading_time = 0
        gpu_time = 0
        before_op_time = time.time()

        train_loss = None

        for batch_idx, inputs in enumerate(self.train_loader):
            for key, ipt in inputs.items():
                if key in ['cam_model', 'pad_info']:
                    inputs[key] = [item.cuda() for item in inputs[key]]
                else:
                    inputs[key] = ipt.cuda()

            data_loading_time += (time.time() - before_op_time)
            before_op_time = time.time()
            
            losses, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.train.dataset.name)
            losses['loss'] = losses['loss'].mean()

            if batch_idx % self.cfg.train.accumulate_gradient == 0:
                self.optimizer.zero_grad() 
            losses['loss'] = losses['loss'] / self.cfg.train.accumulate_gradient
            losses['loss'].backward()
            if batch_idx % self.cfg.train.accumulate_gradient == 0:
                self.optimizer.step()

            duration = time.time() - before_op_time
            gpu_time += duration

            is_summary = (batch_idx % self.cfg.log.train_freq == 0)

            if is_summary:
                self.log_time(batch_idx, duration, losses, data_loading_time, gpu_time)
                self.log_train(inputs, outputs, losses)
                data_loading_time = 0
                gpu_time = 0

            self.iter_step += 1

            if train_loss is None:
                train_loss = {loss_type: float(losses[loss_type].data.mean()) for loss_type in losses}
            else:
                for loss_type in losses:
                    train_loss[loss_type] += float(losses[loss_type].data.mean())
            
            torch.cuda.empty_cache()
            del losses, outputs, inputs
        
        for key in train_loss:
            train_loss[key] /= len(self.train_loader)
        self.logger.info('\t=> Train loss: {}'.format(train_loss))
    
    def test(self):
        self.set_eval()
        time_total = 0
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.test_loader):
                for key, ipt in inputs.items():
                    inputs[key] = ipt.cuda() 
                start_time = time.time()               
                outputs = self.net(inputs, is_test=True)
                torch.cuda.synchronize()
                time_one = time.time() - start_time
                time_total = time_total + time_one
                depth = torch.squeeze(outputs['depth'][-1].data.cpu()).numpy()
                depth = (depth * 256.0).astype('uint16')
                depth_buffer = depth.tobytes()
                depthsave = Image.new("I", depth.T.shape)
                depthsave.frombytes(depth_buffer, 'raw', "I;16")

                save(inputs['sparse'].cpu()*256, './save/sparse.png')
                # depthsave.save('./save_kitti/{:0>10d}.png'.format(batch_idx))
                self.logger.info('************** {} {}'.format(batch_idx, time_one))
                # import matplotlib.pyplot as plt
                # import matplotlib 
                # matplotlib.use('pdf')
                # depth = outputs['depth_1'].cpu().squeeze(0).squeeze(0).numpy()
                # plt.imshow(depth, cmap="jet") 
                # plt.axis('off')
                # plt.savefig("save/%04d" % batch_idx + ".png", bbox_inches = 'tight', pad_inches = 0)
            self.logger.info('************** average time {}'.format(time_total/len(self.test_loader)))
        del inputs, outputs
    
    def save_temp(self, output, input, batch, data, scale=10):
        # save_depth(input["target"].cpu(), './save/'+data+'_{:0>4d}_gt.png'.format(batch))
        # save_depth(input["sparse_noise"].cpu(), './save/'+data+'_{:0>4d}_noise.png'.format(batch))
        # print(input["target"].min(), input["target"].max())
        # print(input["sparse_noise"].min(), input["sparse_noise"].max())
        # return
        # if batch < 90:
        #     return
        # from sklearn.decomposition import PCA
        # import PIL.Image as Image
        import matplotlib.pyplot as plt
        import matplotlib 
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        matplotlib.use('pdf')

        sparse = input["sparse"].cpu()
        mask = (sparse > 0).float()
        conv = torch.nn.Conv2d(1, 1, (3, 3), stride=1, padding=1, bias=False, padding_mode='zeros')
        conv.weight.data = torch.FloatTensor([[[[1, 1, 1],
                                                [1, 1, 1],
                                                [1, 1, 1],]]])
        sparse = conv(sparse)
        mask = conv(mask)
        sparse = sparse/(mask+1e-5)
        mask = mask > 0
        sparse = save_depth(sparse, './save/'+data+'_{:0>4d}_sparse.png'.format(batch))

        mask = mask.float()
        sparse_rgb = input['rgb_s'].cpu()*(1-mask) + sparse * mask
        # save_rgb(input['rgb_s'].cpu(), './save/'+data+'_{:0>4d}_rgb.png'.format(batch))

        # plt.imshow(input['rgb_s'].cpu().squeeze().permute(1,2,0)) 
        plt.imshow(sparse_rgb.squeeze().permute(1,2,0)) #, alpha=0.5) 
        plt.axis('off')
        plt.savefig('./save/'+data+'_{:0>4d}_rgb.png'.format(batch), bbox_inches = 'tight', pad_inches = 0, dpi=100)

        
        save_depth(output['depth'][-1].cpu(), './save/'+data+'_{:0>4d}_d.png'.format(batch))
        from  torchvision import utils as vutils
        err0 = depth_error_image_func(output['depth'][-1].squeeze(1)*scale, input['target'].squeeze(1)*scale)
        vutils.save_image(err0, './save/{:0>4d}_error.png'.format(batch))

        # save_depth_kitti(output['depth'][-1], './save/'+data+'_{:0>4d}_pol_pc.png'.format(batch))

        # save_rgb(input['rgb_s'], './save/'+data+'_{:0>4d}_rgb.png'.format(batch))

        # save_depth(output['depth'][0].cpu(), './save/'+data+'_{:0>4d}_3D->2D.png'.format(batch))
        # save_depth(output['depth'][1].cpu(), './save/'+data+'_{:0>4d}_2D.png'.format(batch))
        # save_depth(output['depth'][2].cpu(), './save/'+data+'_{:0>4d}_3D+2D.png'.format(batch))
        # save_depth(input['target'].cpu(), './save/'+data+'_{:0>4d}_GT.png'.format(batch))
        # save_depth(output['depth'][3].cpu(), './save/'+data+'_{:0>4d}_refine.png'.format(batch))
        
       
        # plt.imshow(output['depth'][2].cpu().squeeze(0).squeeze(0), cmap="jet") 
        # plt.axis('off')
        # plt.savefig('./save/'+data+'_{:0>4d}_ip.png'.format(batch), bbox_inches = 'tight', pad_inches = 0, dpi=100)
        
        # plt.imshow(output['segment'].cpu().squeeze(0).squeeze(0), cmap="rainbow") 
        # plt.axis('off')
        # plt.savefig('./save/'+data+'_{:0>4d}_segment.png'.format(batch), bbox_inches = 'tight', pad_inches = 0, dpi=100)
        
        # from  torchvision import utils as vutils
        # vutils.save_image(input['rgb'].cpu(), './save/{:0>4d}_rgb.png'.format(batch), normalize=True)
        # err0 = depth_error_image_func(output['depth'][0].squeeze(1), input['target'].squeeze(1))
        # err1 = depth_error_image_func(output['depth'][1].squeeze(1), input['target'].squeeze(1))
        # err2 = depth_error_image_func(output['depth'][2].squeeze(1), input['target'].squeeze(1))

        # vutils.save_image(err0, './save/{:0>4d}_error_mde.png'.format(batch))
        # vutils.save_image(err1, './save/{:0>4d}_error_cpm.png'.format(batch))
        # vutils.save_image(err2, './save/{:0>4d}_error_spn.png'.format(batch))

    def val_nyu(self):
        self.logger.info("Val NYU epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.nyu_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda()
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.NYUv2.name)
                # self.save_temp(outputs, inputs, batch_idx, self.cfg.NYUv2.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))
                    self.log_val(inputs, outputs, errors, self.cfg.NYUv2.name)
                
                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.nyu_loader)) 
            
            self.logger.info('\t=> Val NYU error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_nyu)

        del inputs, outputs, errors

        self.set_train()
    
    def val_diodei(self):
        self.logger.info("Val DIODE indoor epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.diodei_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda()
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.DIODEi.name)
                self.save_temp(outputs, inputs, batch_idx, self.cfg.DIODEi.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))
                
                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.diodei_loader)) 
            
            self.logger.info('\t=> Val DIODE indoor error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_diodei)

        del inputs, outputs, errors

        self.set_train()
    
    def val_sunrgbd(self):
        self.logger.info("Val SUNRGBD epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.sunrgbd_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda()

                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.SUNRGBD.name)
                self.save_temp(outputs, inputs, batch_idx, self.cfg.SUNRGBD.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))
                
                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    if key != "sun_num":
                        val_error[key][i] /= (len(self.sunrgbd_loader))
            
            self.logger.info('\t=> Val SUNRGBD error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_sunrgbd)

        del inputs, outputs, errors

        self.set_train()
    
    def val_scannet(self):
        self.logger.info("Val SCANNET epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.scannet_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda()
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.SCANNET.name)
                self.save_temp(outputs, inputs, batch_idx, self.cfg.SCANNET.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))

                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.scannet_loader))
            
            self.logger.info('\t=> Val SCANNET error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_scannet)

        del inputs, outputs, errors

        self.set_train()
    
    def val_middlebury(self):
        self.logger.info("Val MIDDLEBURY epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.middlebury_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda()
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.MIDDLEBURY.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))
   
                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.middlebury_loader))
            
            self.logger.info('\t=> Val MIDDLEBURY error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_middlebury)

        del inputs, outputs, errors

        self.set_train()

    def val_hypersim(self):
        self.logger.info("Val HYPERSIM epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.hypersim_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda()
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.HYPERSIM.name)
                self.save_temp(outputs, inputs, batch_idx, self.cfg.HYPERSIM.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))

                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.hypersim_loader))
            
            self.logger.info('\t=> Val HYPERSIM error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_hypersim)

        del inputs, outputs, errors

        self.set_train()

    def val_eth3d(self):
        self.logger.info("Val ETH3D epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.eth3d_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda()
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.ETH3D.name)
                self.save_temp(outputs, inputs, batch_idx, self.cfg.ETH3D.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))

                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.eth3d_loader)) 
            
            self.logger.info('\t=> Val ETH3D error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_eth3d)

        del inputs, outputs, errors

        self.set_train()
    
    def val_void(self):
        self.logger.info("Val VOID epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.void_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda()
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.VOID1500.name)
                self.save_temp(outputs, inputs, batch_idx, self.cfg.VOID1500.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))

                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.void_loader)) 
            
            self.logger.info('\t=> Val VOID error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_void)

        del inputs, outputs, errors

        self.set_train()
    
    def val_kitti(self):
        self.logger.info("Val KITTI epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.kitti_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda()
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.KITTI.name)
                self.save_temp(outputs, inputs, batch_idx, self.cfg.KITTI.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))
                    self.log_val(inputs, outputs, errors, self.cfg.KITTI.name)
                
                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.kitti_loader)) 
            
            self.logger.info('\t=> Val KITTI error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_kitti)

        del inputs, outputs, errors

        self.set_train()

    def val_ds(self):
        self.logger.info("Val DrivingStereo epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.ds_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda()

                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.DrivingStereo.name)
                self.save_temp(outputs, inputs, batch_idx, self.cfg.DrivingStereo.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))

                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.ds_loader)) 
            
            self.logger.info('\t=> Val DrivingStereo error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_ds)

        del inputs, outputs, errors

        self.set_train()
    
    def val_diodeo(self):
        self.logger.info("Val DIODE outdoor epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.diodeo_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda()

                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.DIODEo.name)
                self.save_temp(outputs, inputs, batch_idx, self.cfg.DIODEo.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))

                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.diodeo_loader)) 
            
            self.logger.info('\t=> Val DIODE outdoor error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_diodeo)

        del inputs, outputs, errors

        self.set_train()
    
    def val_argoverse(self):
        self.logger.info("Val ARGOVERSE epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.arg_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda()
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.ARGOVERSE.name)
                # self.save_temp(outputs, inputs, batch_idx, self.cfg.ARGOVERSE.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))

                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.arg_loader)) 
            
            self.logger.info('\t=> Val ARGOVERSE error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_arg)

        del inputs, outputs, errors

        self.set_train()

    def val_vkitti2(self):
        self.logger.info("Val VKITTI2 epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.vkitti2_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda()
                
                errors, pred = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.VKITTI2.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))
               
                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.vkitti2_loader)) 
            
            self.logger.info('\t=> Val VKITTI2 error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_vkitti2)

        del inputs, errors

    def val_dimli(self):
        self.logger.info("Val DIMLi epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.dimli_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda() 
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.DIMLi.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))

                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.dimli_loader)) 
            
            self.logger.info('\t=> Val DIMLi error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_dimli)

        del inputs, outputs, errors

        self.set_train()
    
    def val_cityscape(self):
        self.logger.info("Val Cityscape epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.cityscape_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda() 
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.Cityscape.name)
                self.save_temp(outputs, inputs, batch_idx, self.cfg.Cityscape.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))

                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.cityscape_loader)) 
            
            self.logger.info('\t=> Val Cityscape error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_cityscape)

        del inputs, outputs, errors

        self.set_train()
    
    def val_tofdc(self):
        self.logger.info("Val TOFDC epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.tofdc_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda() 
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.TOFDC.name)
                self.save_temp(outputs, inputs, batch_idx, self.cfg.TOFDC.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))

                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.tofdc_loader)) 
            
            self.logger.info('\t=> Val TOFDC error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_tofdc)

        del inputs, outputs, errors

        self.set_train()
    
    def val_hammer(self):
        self.logger.info("Val HAMMER epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.hammer_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda()
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.HAMMER.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))

                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.hammer_loader)) 
            
            self.logger.info('\t=> Val HAMMER error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_hammer)

        del inputs, outputs, errors

        self.set_train()
    
    def val_stanford(self):
        self.logger.info("Val Stanford epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.stanford_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda() 
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.Stanford.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))

                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.stanford_loader)) 
            
            self.logger.info('\t=> Val Stanford error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_stanford)

        del inputs, outputs, errors

        self.set_train()

    def val_kitti360(self):
        self.logger.info("Val KITTI360 epoch: {}".format(self.epoch))

        self.set_eval()
        val_error = None
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.kitti360_loader):
                for key, ipt in inputs.items():
                    if key in ['cam_model', 'pad_info']:
                        inputs[key] = [item.cuda() for item in inputs[key]]
                    else:
                        inputs[key] = ipt.cuda() 
                
                errors, outputs = self.net(inputs=inputs, is_test=False, epoch=self.epoch, data=self.cfg.KITTI360.name)

                if val_error is None:
                    val_error = {}
                    for err_type in errors:
                        val_error[err_type] = []
                        for e in errors[err_type]:
                            val_error[err_type].append(float(e.detach().cpu().numpy().mean())) 
                else:
                    for err_type in errors:
                        for e in range(len(errors[err_type])):
                            val_error[err_type][e] += float(errors[err_type][e].detach().cpu().numpy().mean())
                
                is_summary = (batch_idx % self.cfg.log.train_freq == 0)
                if is_summary:
                    self.logger.info("\t  ==> " + str(batch_idx)+"\t" + str({item: round(float(errors[item][-1]), 6) for item in errors}))

                torch.cuda.empty_cache()
            
            for key in val_error:
                for i in range(len(val_error[key])):
                    val_error[key][i] /= (len(self.kitti360_loader)) 
            
            self.logger.info('\t=> Val KITTI360 error: {}'.format(val_error))
            self.save_csv(val_error, self.csvfile_kitti360)

        del inputs, outputs, errors

        self.set_train()
    
    def log_time(self, batch_idx, duration, losses, data_time, gpu_time):
        samples_per_sec = self.cfg.train.batch_size / duration
        time_sofar = time.time() - self.start_time
        print_string = "\t ==> epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time left: {} | CPU/GPU time: {:0.1f}s/{:0.1f}s"
        self.logger.info(print_string.format(self.epoch, batch_idx, samples_per_sec, losses['loss'].data.cpu().mean(),
                                  sec_to_hm_str(time_sofar), data_time, gpu_time))
        self.logger.info("\t ==> " + str({item: round(float(losses[item].data.cpu().mean()), 6) for item in losses}))
    
    def log_loss(self, losses, mode):
        writer = self.writers[mode]
        for l, v in losses.items():
            for e in range(len(v)):
                writer.add_scalar("epoch_{}_{}".format(l, e), v[e], self.iter_step)

    def log_train(self, inputs, outputs, losses):
        writer = self.writers['train']
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v.mean(), self.iter_step)

        for j in range(self.cfg.train.batch_size):
            writer.add_image('rgb', inputs['rgb'][j], self.iter_step)
            # writer.add_image('sparse', inputs['sparse'][j], self.iter_step)
            writer.add_image('target', inputs['target'][j], self.iter_step)

            for idx, depth in enumerate(outputs['depth']):
                depth = depth.data
                writer.add_image('depth_'+str(idx), normalize_image(depth[j]), self.iter_step)
                writer.add_image('error_'+str(idx), depth_error_image_func(depth[j].squeeze(1), inputs['target'][j]), self.iter_step)
                
            break
                
    def log_val(self, inputs, outputs, losses, data):
        writer = self.writers['val']
        for l, v in losses.items():
            for e in range(len(v)):
                writer.add_scalar("{}_{}".format(l, e), v[e], self.iter_step)

        for j in range(1):
            writer.add_image(data + '_rgb', inputs['rgb'][j], self.iter_step)
            writer.add_image(data + '_sparse', inputs['sparse'][j], self.iter_step)
            writer.add_image(data + '_target', inputs['target'][j], self.iter_step)

            for idx, depth in enumerate(outputs['depth']):
                depth = depth.data
                writer.add_image(data + '_depth_'+str(idx), normalize_image(depth[j]), self.iter_step)
                writer.add_image(data + '_error_'+str(idx), depth_error_image_func(depth[j].squeeze(1), inputs['target'][j]), self.iter_step)
    
    def save_model(self):
        ckpt_data = {'epoch': self.epoch, 'model': self.net.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(ckpt_data, "{}/checkpoint_{:0>6}.ckpt".format(self.log_path, self.epoch))

    def load_model(self):
        self.logger.info("Loading model {}".format(self.cfg.load_ckpt))
        pre_state_dict = torch.load(self.cfg.load_ckpt)
        model_state_dict = pre_state_dict['model']
        dict_no_module = {}
        for k, v in model_state_dict.items():
            if 'module' in k :
                dict_no_module[k[7:]] = v
            else:
                dict_no_module[k] = v
        key_m, key_u = self.net.load_state_dict(dict_no_module)

        if key_u:
            self.log('Unexpected keys :')
            self.log(key_u)

        if key_m:
            self.log('Missing keys :')
            self.log(key_m)

        self.net.load_state_dict(dict_no_module) # pre_state_dict['model'])

        self.optimizer.load_state_dict(pre_state_dict['optimizer'])
        self.start_epoch = pre_state_dict['epoch'] + 1

        # state_dict = {}
        # model_dict = self.net.module.state_dict()
        # for k, v in model_dict.items():
        #     if k[:12] in ['total_offset']:
        #         continue
        #     state_dict[k] = pre_state_dict['model'][k]
        # model_dict.update(state_dict)

        # self.net.module.load_state_dict(model_dict)

def fix_bn(model):
    classname = model.__class__.__name__
    if classname.find('BatchNorm') != -1:
        model.eval()

def release_bn(model):
    classname = model.__class__.__name__
    if classname.find('BatchNorm') != -1:
        model.train()