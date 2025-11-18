import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
       
class Conv2D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, \
        dilation=1, groups=1, bias=True, activation='relu', norm='batch', padding_mode='reflect'):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride=stride, padding=padding, \
            dilation=dilation, groups=groups, padding_mode=padding_mode, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)
        elif self.norm == 'layernorm':
            self.bn = nn.LayerNorm2d(output_size)
        elif self.norm == 'rezero':
            if input_size==output_size and stride==1:
                self.alpha = nn.Parameter(torch.tensor(0.0))
            else:
                self.alpha = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'elu':
            self.act = nn.ELU(inplace=True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif self.activation == 'softmax':
            self.act = nn.Softmax(dim=1)
        elif self.activation == 'gelu':
            self.act = nn.GELU()
        
        torch.nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        if self.norm is not None:
            if self.norm  == 'rezero':
                out = self.conv(x)
                if self.alpha is not None:
                    out = self.alpha * out + x
            else:
                out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class Deconv2D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, \
        output_padding=0, groups=1, bias=True, activation='relu', norm='batch'):
        super(Deconv2D, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride=stride, padding=padding, \
            output_padding=output_padding, groups=groups, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)
        elif self.norm == 'layernorm':
            self.bn = nn.LayerNorm2d(output_size)
        elif self.norm == 'rezero':
            self.norm = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'elu':
            self.act = nn.ELU(inplace=True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif self.activation == 'gelu':
            self.act = nn.GELU()
        
        torch.nn.init.kaiming_normal_(self.deconv.weight)

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=2, downsample=None, activation='relu', norm='batch'):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2D(inplanes, planes, kernel_size=3, stride=stride, padding=1, \
            dilation=1, activation=activation, norm=norm)
        self.conv2 = Conv2D(planes, planes, kernel_size=3, stride=1, padding=1, \
            dilation=1, activation=None, norm=norm)

        if downsample is None:
            if stride != 1 or inplanes != planes * BasicBlock.expansion:
                downsample = Conv2D(inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, padding=0,\
                   activation=None, norm=norm)
        
        self.downsample = downsample
       
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride),
                     padding=(1, 1), bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, (1, 1), bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, (1, 1), bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, (kernel_size, kernel_size), padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlockReZero(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, ratio=16):
        super(BasicBlockReZero, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.ca = ChannelAttention(planes, ratio=ratio)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.alpha * out + x

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

