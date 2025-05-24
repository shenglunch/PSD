import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np

class Interpolate(nn.Module):
    def __init__(self, mode = 'bilinear', scale_factor=2, size=None, align_corners = False):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.size = size
    
    def forward(self, x):
        if self.size is not None:
            return F.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)

        if self.scale_factor > 1:
            return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        else:
            h = int(self.scale_factor*x.shape[2]) 
            w = int(self.scale_factor*x.shape[3])
            return F.interpolate(x, size=(h, w), mode=self.mode, align_corners=self.align_corners)
            
def subdivision(depth, similarity0, similarity1, similarity2, similarity3, similarity4):
    with torch.no_grad():
        tmp1 = (similarity1 - similarity3)
        tmp2 = (similarity1 - similarity2 - similarity2 + similarity3)
        tmp3 = tmp1 - (similarity2 - similarity4)
        tmp4 = (similarity0 - similarity2) - tmp1
        rate = torch.zeros_like(depth, dtype=depth.dtype, device=depth.device)
        mask = ((tmp1 > 0) & (tmp2 > 0)) | ((tmp1 < 0)  & (tmp2 < 0))
        rate_temp = ((tmp1 / (tmp2 + tmp2+1e-8)) +  (tmp1 /   (tmp3+1e-8))) / 2.
        rate[mask]=rate_temp[mask]
        rate_temp = (( tmp1 / (tmp2 + tmp2+1e-8)) + ( tmp1 / (tmp4+1e-8))) / 2.
        mask = ~mask
        rate[mask] = rate_temp[mask]

    return rate

def predict_depth(similarity, min_depth, num_depth, step): 
    with torch.no_grad():
        index = similarity.argmax(1, keepdim=True)
        # five point
        index_1 = index - 1
        index_1 = torch.where(index_1 < 0, torch.full_like(index_1, 0), index_1)
        index_2 = index - 2
        index_2 = torch.where(index_2 < 0, torch.full_like(index_2, 0), index_2)
        index1 = index + 1
        index1 = torch.where(index1 > (num_depth-1), torch.full_like(index1, (num_depth-1)), index1)
        index2 = index + 2
        index2 = torch.where(index2 > (num_depth-1), torch.full_like(index2, (num_depth-1)), index2)
        # five probability
        pro_1 = torch.gather(similarity, 1, index_1)
        pro_2 = torch.gather(similarity, 1, index_2)
        pro   = torch.gather(similarity, 1, index)
        pro1  = torch.gather(similarity, 1, index1)
        pro2  = torch.gather(similarity, 1, index2)
        # subdivision
        similarity1 = 1. - pro_2
        similarity2 = 1. - pro_1
        similarity3 = 1. - pro
        similarity4 = 1. - pro1
        similarity5 = 1. - pro2
        tmp1 = (similarity2 - similarity4)
        tmp2 = (similarity2 - similarity3 - similarity3 + similarity4)
        tmp3 = tmp1 - (similarity3 - similarity5)
        tmp4 = (similarity1 - similarity3) - tmp1

        rate = torch.zeros_like(index).cuda().float()
        mask = ((tmp1 > 0) & (tmp2 > 0)) | ((tmp1 < 0)  & (tmp2 < 0))
        rate_temp = ((tmp1 / (tmp2 + tmp2+1e-8)) +  (tmp1 /   (tmp3+1e-8))) / 2.
        rate[mask]=rate_temp[mask]
        rate_temp = (( tmp1 / (tmp2 + tmp2+1e-8)) + ( tmp1 / (tmp4+1e-8))) / 2.
        mask = ~mask
        rate[mask] = rate_temp[mask]

        depth = (index+rate)*step + min_depth

    return depth

def interpolate(x, scale=1, mode="nearest"):
    return F.interpolate(x, scale_factor=scale, mode=mode)

class MaxPool2D(nn.Module):
    def __init__(self, kernel_size=3, stride=3, padding=0):
        super(MaxPool2D, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        out = self.pool(x)
        return out

class MinPool2D(nn.Module):
    def __init__(self, kernel_size=3, stride=3, padding=0):
        super(MinPool2D, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        out = self.pool(-x)
        return -out

class MeanPool2D(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super(MeanPool2D, self).__init__()

        self.pool = nn.Conv2d(in_channels=dim, out_channels=dim,
                            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        weight = torch.ones(1, 1, kernel_size, kernel_size)
        self.pool.weight = nn.Parameter(weight)
        for param in self.pool.parameters():
            param.requires_grad = False
        self.div = kernel_size*kernel_size
    
    def forward(self, x):
        out = self.pool(x) / self.div
        return out

class Conv1D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=1, stride=1, padding=0, \
        dilation=1, groups=1, bias=True, activation='relu', norm='batch', padding_mode='reflect', L=0):
        super(Conv1D, self).__init__()
        self.conv = nn.Conv1d(input_size, output_size, kernel_size, stride=stride, padding= dilation if dilation > 1 else padding, \
            dilation=dilation, groups=groups, padding_mode=padding_mode, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm1d(output_size)
        elif self.norm == 'layer':
            self.bn = nn.LayerNorm(L) #nn.LayerNorm(L)?!
        elif self.norm == 'group':
            self.bn = nn.GroupNorm(input_size, output_size)

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
        
        torch.nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

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

class Conv3D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, \
        dilation=1, groups=1, bias=True, activation='relu', norm='batch', padding_mode='reflect'):
        super(Conv3D, self).__init__()
        self.conv = nn.Conv3d(input_size, output_size, kernel_size, stride=stride, padding=padding, \
            dilation=dilation, groups=groups, padding_mode=padding_mode, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm3d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm3d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'relu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'elu':
            self.act = nn.ELU(inplace=True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()
        
        torch.nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class Deconv3D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, \
        output_padding=1, groups=1, bias=True, activation='relu', norm='batch'):
        super(Deconv3D, self).__init__()
        self.deconv = nn.ConvTranspose3d(input_size, output_size, kernel_size, stride=stride, padding=padding, \
        output_padding=output_padding, groups=groups, bias=True)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm3d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm3d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'relu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'elu':
            self.act = nn.ELU(inplace=True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()
        
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

class ShuffleBlock(nn.Module):
    def __init__(self, inp, oup, stride=1, benchmodel=2, activation='relu', norm='batch'):
        super(ShuffleBlock, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        
        if self.benchmodel == 1:
        	self.banch2 = nn.Sequential(
                Conv2D(oup_inc, oup_inc, kernel_size=1, stride=1, padding=0, \
                    activation=activation, norm=norm), # pw
                Conv2D(oup_inc, oup_inc, kernel_size=3, stride=stride, padding=1, \
                    groups=oup_inc, activation=None, norm=norm), # dw
                Conv2D(oup_inc, oup_inc, kernel_size=1, stride=1, padding=0, \
                    activation=activation, norm=norm), # pw-linear
            )                
        else:                  
            self.banch1 = nn.Sequential(
                Conv2D(inp, inp, kernel_size=3, stride=stride, padding=1, \
                    groups=inp, activation=None, norm=norm), # dw
                Conv2D(inp, oup_inc, kernel_size=1, stride=1, padding=0, \
                    activation=activation, norm=norm), # pw-linea
            )       
            self.banch2 = nn.Sequential(
                Conv2D(inp, oup_inc, kernel_size=1, stride=1, padding=0, 
                    activation=activation, norm=norm), # pw
                Conv2D(oup_inc, oup_inc, kernel_size=3, stride=stride, padding=1, 
                    groups=oup_inc, activation=None, norm=norm), # dw
                Conv2D(oup_inc, oup_inc, kernel_size=1, stride=1, padding=0, 
                    activation=activation, norm=norm), # pw-linea
            )     
    
    def channel_shuffle(self, x, groups):
        b, c, h, w = x.data.size()
        channels_per_group = c // groups 
        # reshape
        x = x.view(b, groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(b, -1, h, w)
        return x
        
    def forward(self, x):
        if self.benchmodel == 1:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            x2 = self.banch2(x2)
            out = torch.cat((x1, x2), dim=1)
        elif self.benchmodel == 2:
            x1 = self.banch1(x)
            x2 = self.banch2(x)
            out = torch.cat((x1, x2), dim=1)
        
        out = self.channel_shuffle(out, 2)
        return out

class SEBlock(nn.Module):
    def __init__(self, inplanes, scale=16):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=inplanes, out_features=round(inplanes / scale)),
            nn.ReLU(True))
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=round(inplanes / scale), out_features=inplanes),
            nn.Sigmoid())

    def forward(self, x): 
        b, c = x.size(0), x.size(1)
        x1 = self.gap(x).view(b, -1)
        x1 = self.fc1(x1)
        s = self.fc2(x1).view(b, c, 1, 1)
        y = x + x*s
        return y

class GRUGate(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GRUGate, self).__init__()

        self.convz1 = Conv1D(in_ch, out_ch, kernel_size=1, stride=1, padding=0, activation='sigmoid', norm=None)
        self.convr1 = Conv1D(in_ch, out_ch, kernel_size=1, stride=1, padding=0, activation='sigmoid', norm=None)
        self.convq1 = Conv1D(in_ch, out_ch, kernel_size=1, stride=1, padding=0, activation='tanh', norm=None)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1) 
        z = self.convz1(hx)
        r = self.convr1(hx)
        q = self.convq1(torch.cat([r * h, x], dim=1))
        h = (1 - z) * h + z * q
        return h

class CSPNGenerateAccelerate(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerateAccelerate, self).__init__()
        self.kernel_size = kernel_size
        self.generate = Conv2D(in_channels, self.kernel_size * self.kernel_size - 1, \
                                kernel_size=3, stride=1, padding=1, activation=None, norm='batch')

    def forward(self, feature):

        guide = self.generate(feature)

        #normalization in standard CSPN
        #'''
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)
        #'''
        #weight_pad = [i for i in range(self.kernel_size * self.kernel_size)]

        half1, half2 = torch.chunk(guide, 2, dim=1)
        output =  torch.cat((half1, guide_mid, half2), dim=1)
        return output

class CSPNAccelerate(nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=1, stride=1):
        super(CSPNAccelerate, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, kernel, input, input0): #with standard CSPN, an addition input0 port is added
        bs = input.size()[0]
        h, w = input.size()[2], input.size()[3]
        input_im2col = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        kernel = kernel.reshape(bs, self.kernel_size * self.kernel_size, h * w)

        # standard CSPN
        input0 = input0.view(bs, 1, h * w)
        mid_index = int((self.kernel_size*self.kernel_size-1)/2)
        input_im2col[:, mid_index:mid_index+1, :] = input0

        #print(input_im2col.size(), kernel.size())
        output = torch.einsum('ijk,ijk->ik', (input_im2col, kernel))
        return output.view(bs, 1, h, w)

class SparseDownSampleClose(nn.Module):
    def __init__(self, stride):
        super(SparseDownSampleClose, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600
    def forward(self, d, mask):
        encode_d = - (1-mask)*self.large_number - d

        d = - self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1-mask_result)*self.large_number

        return d_result, mask_result

class Filter(nn.Module):
    def __init__(self, threshold=1.0):
        super(Filter, self).__init__()
        self.filter = torch.nn.Conv2d(1, 1, (7, 7), stride=1, padding=3, bias=True, padding_mode='zeros')
        self.filter.weight.data = torch.FloatTensor([[[[1, 1, 1, 1, 1, 1, 1],
                                                       [1, 1, 1, 1, 1, 1, 1],
                                                       [1, 1, 1, 1, 1, 1, 1],
                                                       [1, 1, 1, 1, 1, 1, 1],
                                                       [1, 1, 1, 1, 1, 1, 1],
                                                       [1, 1, 1, 1, 1, 1, 1],
                                                       [1, 1, 1, 1, 1, 1, 1]]]])
        
        self.threshold = threshold
    
    def forward(self, sparse, mask):
        sparse_sum = self.filter(sparse)
        mask_count = self.filter(mask.float())
        sparse_avg = sparse_sum / (mask_count+1e-5)
        sparse_avg = sparse_avg * mask
        inlier = (sparse - sparse_avg) < self.threshold
        sparse = sparse * inlier

        return sparse.detach(), inlier.detach()

class SPN(nn.Module):
    def __init__(self):
        super(SPN, self).__init__()
        self.filter = Filter()

        self.proj1 = Conv2D(32, 32, kernel_size=1, stride=1, padding=0, activation=None, norm='batch')
        self.kernel_conf_layer = Conv2D(32, 3, kernel_size=3, stride=1, padding=1, activation=None, norm='batch')
        self.mask_layer = Conv2D(32, 1, kernel_size=3, stride=1, padding=1, activation=None, norm='batch')
        self.iter_guide_layer3 = CSPNGenerateAccelerate(32, 3)
        self.iter_guide_layer5 = CSPNGenerateAccelerate(32, 5)
        self.iter_guide_layer7 = CSPNGenerateAccelerate(32, 7)

        self.proj2 = Conv2D(64, 64, kernel_size=1, stride=1, padding=0, activation=None, norm='batch')
        self.kernel_conf_layer_s2 = Conv2D(64, 3, kernel_size=3, stride=1, padding=1, activation=None, norm='batch')
        self.mask_layer_s2 = Conv2D(64, 1, kernel_size=3, stride=1, padding=1, activation=None, norm='batch')
        self.iter_guide_layer3_s2 = CSPNGenerateAccelerate(64, 3)
        self.iter_guide_layer5_s2 = CSPNGenerateAccelerate(64, 5)
        self.iter_guide_layer7_s2 = CSPNGenerateAccelerate(64, 7)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.nnupsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.downsample = SparseDownSampleClose(stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.CSPN3 = CSPNAccelerate(kernel_size=3, dilation=1, padding=1, stride=1)
        self.CSPN5 = CSPNAccelerate(kernel_size=5, dilation=1, padding=2, stride=1)
        self.CSPN7 = CSPNAccelerate(kernel_size=7, dilation=1, padding=3, stride=1)
        self.CSPN3_s2 = CSPNAccelerate(kernel_size=3, dilation=2, padding=2, stride=1)
        self.CSPN5_s2 = CSPNAccelerate(kernel_size=5, dilation=2, padding=4, stride=1)
        self.CSPN7_s2 = CSPNAccelerate(kernel_size=7, dilation=2, padding=6, stride=1)

        # CSPN
        ks = 3
        encoder3 = torch.zeros(ks * ks, ks * ks, ks, ks)#.cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder3[index] = 1
        self.encoder3 = nn.Parameter(encoder3, requires_grad=False)

        ks = 5
        encoder5 = torch.zeros(ks * ks, ks * ks, ks, ks)#.cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder5[index] = 1
        self.encoder5 = nn.Parameter(encoder5, requires_grad=False)

        ks = 7
        encoder7 = torch.zeros(ks * ks, ks * ks, ks, ks)#.cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder7[index] = 1
        self.encoder7 = nn.Parameter(encoder7, requires_grad=False)
    
    def forward(self, feature1, feature2, depth, sparse):
        sparse, inlier1 = self.filter(sparse)
        mask = sparse > 0
        residual = (sparse - depth) * mask
        inlier2 = residual < 1.0
        sparse = sparse * inlier2

        feature1 = self.proj1(feature1)
        feature2 = self.proj2(feature2)
        valid_mask = torch.where(sparse>0, torch.full_like(sparse, 1.0), torch.full_like(sparse, 0.0))
        
        d_s2, valid_mask_s2 = self.downsample(sparse, valid_mask)
        mask_s2 = self.mask_layer_s2(feature2)
        mask_s2 = torch.sigmoid(mask_s2)
        mask_s2 = mask_s2*valid_mask_s2

        kernel_conf_s2 = self.kernel_conf_layer_s2(feature2)
        kernel_conf_s2 = self.softmax(kernel_conf_s2)
        kernel_conf3_s2 = self.nnupsample(kernel_conf_s2[:, 0:1, :, :])
        kernel_conf5_s2 = self.nnupsample(kernel_conf_s2[:, 1:2, :, :])
        kernel_conf7_s2 = self.nnupsample(kernel_conf_s2[:, 2:3, :, :])

        guide3_s2 = self.iter_guide_layer3_s2(feature2)
        guide5_s2 = self.iter_guide_layer5_s2(feature2)
        guide7_s2 = self.iter_guide_layer7_s2(feature2)

        depth_s2 = self.nnupsample(d_s2)
        mask_s2 = self.nnupsample(mask_s2)
        depth3 = depth5 = depth7 = depth

        mask = self.mask_layer(feature1)
        mask = torch.sigmoid(mask)
        mask = mask * valid_mask

        kernel_conf = self.kernel_conf_layer(feature1)
        kernel_conf = self.softmax(kernel_conf)
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        guide3 = self.iter_guide_layer3(feature1)
        guide5 = self.iter_guide_layer5(feature1)
        guide7 = self.iter_guide_layer7(feature1)

        guide3 = kernel_trans(guide3, self.encoder3)
        guide5 = kernel_trans(guide5, self.encoder5)
        guide7 = kernel_trans(guide7, self.encoder7)

        guide3_s2 = kernel_trans(guide3_s2, self.encoder3)
        guide5_s2 = kernel_trans(guide5_s2, self.encoder5)
        guide7_s2 = kernel_trans(guide7_s2, self.encoder7)

        guide3_s2 = self.nnupsample(guide3_s2)
        guide5_s2 = self.nnupsample(guide5_s2)
        guide7_s2 = self.nnupsample(guide7_s2)
    
        for i in range(6):
            depth3 = self.CSPN3_s2(guide3_s2, depth3, depth)
            depth3 = mask_s2*depth_s2 + (1-mask_s2)*depth3
            depth5 = self.CSPN5_s2(guide5_s2, depth5, depth)
            depth5 = mask_s2*depth_s2 + (1-mask_s2)*depth5
            depth7 = self.CSPN7_s2(guide7_s2, depth7, depth)
            depth7 = mask_s2*depth_s2 + (1-mask_s2)*depth7

        depth_s2 = kernel_conf3_s2*depth3 + kernel_conf5_s2*depth5 + kernel_conf7_s2*depth7
        refined_depth_s2 = depth_s2

        depth3 = depth5 = depth7 = refined_depth_s2

        #prop
        for i in range(6):
            depth3 = self.CSPN3(guide3, depth3, depth_s2)
            depth3 = mask*sparse + (1-mask)*depth3
            depth5 = self.CSPN5(guide5, depth5, depth_s2)
            depth5 = mask*sparse + (1-mask)*depth5
            depth7 = self.CSPN7(guide7, depth7, depth_s2)
            depth7 = mask*sparse + (1-mask)*depth7

        refined_depth = kernel_conf3*depth3 + kernel_conf5*depth5 + kernel_conf7*depth7

        return refined_depth, sparse

class CrossAttn(nn.Module):
    def __init__(self, dim):
        super(CrossAttn, self).__init__()

        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            Conv2D(dim, dim, kernel_size=1, stride=1, padding=0, activation='relu', norm=None),
            nn.Sigmoid())
        
        self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            Conv2D(dim, dim, kernel_size=1, stride=1, padding=0, activation='relu', norm=None),
            nn.Sigmoid())

        self.cross_conv = Conv2D(dim*2, dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None)

    def _concat(self, fd, fe, dim=1):
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=False)

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, rgb, depth):
        ca_rgb = self.channel_attention_rgb(self.squeeze_rgb(rgb))
        rgb_s = rgb * ca_rgb.expand_as(rgb)

        ca_depth = self.channel_attention_depth(self.squeeze_depth(depth))
        depth_s = depth * ca_depth.expand_as(ca_depth)

        ca_c = torch.softmax(ca_rgb + ca_depth, dim=1)

        rgb_c = rgb * ca_c.expand_as(rgb)
        depth_c = depth * ca_c.expand_as(depth)

        rgb_f = rgb_c + rgb_s
        depth_f = depth_c + depth_s

        output = self.cross_conv(self._concat(rgb_f, depth_f))

        return output
