# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .conv import DWConv

__all__ = (
    "RepHDW",
    "Combined_Hybrid_Pz_Enhanced",
    "MSGI_FCM",
    "CombinedModule_to",
    "CombinedModule",
    "CombinedModule_1to1",
    "CombinedModule_1to3"

   # "SPDConv",


)


class Conv(nn.Module):
    '''Normal Conv with SiLU activation'''

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class AVG(nn.Module):
    def __init__(self, down_n=2):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
        self.down_n = down_n
        # self.output_size = np.array([H, W])

    def forward(self, x):
        B, C, H, W = x.shape
        H = int(H / self.down_n)
        W = int(W / self.down_n)
        output_size = np.array([H, W])
        x = self.avg_pool(x, output_size)
        return x


class RepHDW(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut=True, expansion=0.5, kersize=5, depth_expansion=1,
                 small_kersize=3, use_depthwise=True):
        super(RepHDW, self).__init__()
        c1 = int(out_channels * expansion) * 2
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m = nn.ModuleList(
            DepthBottleneckUni(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise) for _
            in range(depth))
        self.conv2 = Conv(c_ * (depth + 2), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_), 1))
        for conv in self.m:
            y = conv(x_out[-1])
            x_out.append(y)
        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return y_out


class DepthBottleneckUni(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize=5,
                 expansion_depth=1,
                 small_kersize=3,
                 use_depthwise=True):
        super(DepthBottleneckUni, self).__init__()

        mid_channel = int(in_channels * expansion_depth)
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:

            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, out_channels, kernel_size=1)

        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)

        y = self.act(self.conv2(y))

        y = self.one_conv(y)
        return y


class UniRepLKNetBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 deploy=False,
                 attempt_use_lk_impl=True):
        super().__init__()
        if deploy:
            print('------------------------------- Note: deploy mode')
        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 3:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              attempt_use_lk_impl=attempt_use_lk_impl)
        else:
            assert kernel_size in [3]
            self.dwconv = get_conv2d_uni(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                         dilation=1, groups=dim, bias=deploy,
                                         attempt_use_lk_impl=attempt_use_lk_impl)

        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_bn(dim)

    def forward(self, inputs):

        out = self.norm(self.dwconv(inputs))
        return out

    def reparameterize(self):
        if hasattr(self.dwconv, 'merge_dilated_branches'):
            self.dwconv.merge_dilated_branches()
        if hasattr(self.norm, 'running_var'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.dwconv, 'lk_origin'):
                self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
                self.dwconv.lk_origin.bias.data = self.norm.bias + (
                        self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            else:
                conv = nn.Conv2d(self.dwconv.in_channels, self.dwconv.out_channels, self.dwconv.kernel_size,
                                 self.dwconv.padding, self.dwconv.groups, bias=True)
                conv.weight.data = self.dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1)
                conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std
                self.dwconv = conv
            self.norm = nn.Identity()


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """

    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d_uni(channels, channels, kernel_size, stride=1,
                                        padding=kernel_size // 2, dilation=1, groups=channels, bias=deploy,
                                        )
        self.attempt_use_lk_impl = attempt_use_lk_impl

        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [7, 5, 3]
            self.dilates = [1, 1, 1]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3]
            self.dilates = [1, 1]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]
        elif kernel_size == 3:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]


        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d_uni(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                         padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0), bias=True,
                                         attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


def get_conv2d_uni(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                   attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (
    kernel_size[0] // 2, kernel_size[1] // 2)

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1), dtype=kernel.dtype, device=kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


def get_bn(channels):
    return nn.BatchNorm2d(channels)


class DepthBottleneckUniv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize=5,
                 expansion_depth=1,
                 small_kersize=3,
                 use_depthwise=True):
        super(DepthBottleneckUniv2, self).__init__()

        mid_channel = int(in_channels * expansion_depth)
        mid_channel2 = mid_channel
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, mid_channel2, kernel_size=1)

            self.conv3 = UniRepLKNetBlock(mid_channel2, kernel_size=kersize)
            self.act1 = nn.SiLU()
            self.one_conv2 = Conv(mid_channel2, out_channels, kernel_size=1)
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act(self.conv2(y))
        y = self.one_conv(y)
        y = self.act1(self.conv3(y))
        y = self.one_conv2(y)
        return y


class RepHMS(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5, shortcut=True,
                 expansion=0.5,
                 small_kersize=3, use_depthwise=True):
        super(RepHMS, self).__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([
                DepthBottleneckUniv2(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
                for _ in range(depth)
            ])
            self.RepElanMSBlock.append(DepthBlock)

        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:
                        # cascade = [cascade[-1]]
                        if self.depth > 1:
                            cascade = [cascade[-1]]
                        else:
                            cascade = []
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])

        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out


class DepthBottleneckv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize=5,
                 expansion_depth=1,
                 small_kersize=3,
                 use_depthwise=True):
        super(DepthBottleneckv2, self).__init__()

        mid_channel = int(in_channels * expansion_depth)
        mid_channel2 = mid_channel
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = DWConv(mid_channel, mid_channel, kersize)
            # self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, mid_channel2, kernel_size=1)

            self.conv3 = DWConv(mid_channel2, mid_channel2, kersize)
            # self.act1 = nn.SiLU()
            self.one_conv2 = Conv(mid_channel2, out_channels, kernel_size=1)
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.one_conv(y)
        y = self.conv3(y)
        y = self.one_conv2(y)
        return y


class ConvMS(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5, shortcut=True,
                 expansion=0.5,
                 small_kersize=3, use_depthwise=True):
        super(ConvMS, self).__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([
                DepthBottleneckv2(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
                for _ in range(depth)
            ])
            self.RepElanMSBlock.append(DepthBlock)

        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:
                        # cascade = [cascade[-1]]
                        if self.depth > 1:
                            cascade = [cascade[-1]]
                        else:
                            cascade = []
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])

        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out












class ImprovedFCM(nn.Module):
    def __init__(self, dim,):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        # x33 = self.spatial(x4) * x3
        # x44 = self.channel(x3) * x4
        return x3,x4

# ========================
#   特征重标定模块（替代原乘法交互）
# ========================
class FeatureCalibration(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spatial_att = Spatial(dim)  # 保持原Spatial模块
        self.channel_att = Channel(dim)  # 保持原Channel模块

    def forward(self, x3, x4):
        # 特征重标定（避免直接相乘）
        x3_calib = x3 * self.spatial_att(x4)
        x4_calib = x4 * self.channel_att(x3)
        return x3_calib + x4_calib  # 改为相加融合

    # def forward(self, x):  # 接受单一输入
    #     # 示例：自注意力机制（可以用其他方式）
    #     x_calib = x * self.spatial_att(x)  # 空间注意力
    #     x_calib += x * self.channel_att(x)  # 通道注意力
    #     return x_calib







# 目前效果最好的CombinedModule
class CombinedModule(nn.Module):  
    def __init__(self , dim , dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim

        # === 优化版 FCM Part ===
        self.one = dim - dim // 4
        self.two = dim // 4
        # 优化点3: 减少卷积层数 - 只用1个3x3替代原来的2个3x3，计算量减少约50%
        self.conv1 = Conv(self.one , dim , 3 , 1)  # 直接输出到dim维度
        self.conv2 = Conv(self.two , dim , 1 , 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

        # === Pzconv Part ===
        self.dim = dim_out

        # 分支 1: 3x3
        self.dw_conv3x3 = nn.Conv2d(dim_out , dim_out , kernel_size=3 , padding=1 , groups=dim_out)
        self.bn1 = nn.BatchNorm2d(dim_out)
        self.pw_conv3x3 = nn.Conv2d(dim_out , dim_out , kernel_size=1)
        self.bn2 = nn.BatchNorm2d(dim_out)

        # 分支 2: 分离卷积 (3x1 + 1x3) 替代 5x5，计算量减少约44%
        self.conv_h = nn.Conv2d(dim_out , dim_out , kernel_size=(3 , 1) , padding=(1 , 0) , groups=dim_out)
        self.conv_v = nn.Conv2d(dim_out , dim_out , kernel_size=(1 , 3) , padding=(0 , 1) , groups=dim_out)
        self.bn3 = nn.BatchNorm2d(dim_out)
        self.pw_conv_sep = nn.Conv2d(dim_out , dim_out , kernel_size=1)
        self.bn4 = nn.BatchNorm2d(dim_out)

        # self.dw_conv5x5 = nn.Conv2d(dim_out, dim_out, kernel_size=5, padding=2, groups=dim_out)
        # self.bn3 = nn.BatchNorm2d(dim_out)
        # self.pw_conv_sep = nn.Conv2d(dim_out, dim_out, kernel_size=1)
        # self.bn4 = nn.BatchNorm2d(dim_out)


        # identity
        self.identity = nn.Identity()

        # 最终融合
        self.final_pw_conv = nn.Conv2d(dim_out * 3 , dim_out , kernel_size=1)
        self.bn_final = nn.BatchNorm2d(dim_out)

        # 激活函数
        self.act = nn.ReLU(inplace=True)

    def forward(self , x):
        # === 优化版 FCM Forward ===
        x1 , x2 = torch.split(x , [self.one , self.two] , dim=1)
        # 优化点3: 减少卷积层数 - 只用1个3x3
        x3 = self.conv1(x1)  # 直接输出到dim维度
        x4 = self.conv2(x2)

        # 优化点1&2: 使用优化后的注意力机制
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        fcm_out = x33 + x44

        # === Pzconv Forward（每个分支都严格还原原始流程）===
        y1 = self.dw_conv3x3(fcm_out)
        y1 = self.bn1(y1)
        y1 = self.act(y1)
        y1 = self.pw_conv3x3(y1)
        y1 = self.bn2(y1)
        y1 = self.act(y1)

        y2 = self.conv_h(fcm_out)  # 3x1 卷积
        y2 = self.conv_v(y2)  # 1x3 卷积
        #y2 = self.dw_conv5x5(fcm_out)  # 直接 5x5 depthwise

        y2 = self.bn3(y2)
        y2 = self.act(y2)
        y2 = self.pw_conv_sep(y2)
        y2 = self.bn4(y2)
        y2 = self.act(y2)

        y5 = self.identity(fcm_out)

        z = torch.cat([y1 , y2 , y5] , dim=1)
        pzconv_out = self.act(self.bn_final(self.final_pw_conv(z)))

        return pzconv_out + fcm_out





