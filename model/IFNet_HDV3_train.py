import torch
import torch.nn as nn
import torch.nn.functional as F
# from warplayer import warp #生成onnx用
from model.warplayer import warp   #插帧用 训练用
# from refine import * #生成onnx用
from model.refine import *#插帧用 训练用
#
#####DWConv修改
# class DWConv(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(DWConv, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=True)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
#         self.prelu = nn.PReLU(out_channels)
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.prelu(x)
#         return x

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    # 原版模型
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )
    # return DWConv(in_planes, out_planes, stride=stride)#######DWConv修改

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1)
        )
        self.convblock0 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock1 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock2 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock3 = nn.Sequential(conv(c, c), conv(c, c))

        # 光流分支
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 4, 4, 2, 1)
        )
        # mask 分支
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 1, 4, 2, 1)
        )

    def forward(self, x, flow, scale=1):
        scale = float(scale)
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * (1. / scale)
            x = torch.cat((x, flow), 1)

        feat = self.conv0(x)
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat

        flow_out = self.conv1(feat)
        flow_out = F.interpolate(flow_out, scale_factor=scale, mode="bilinear", align_corners=False) * scale
        mask_out = self.conv2(feat)
        mask_out = F.interpolate(mask_out, scale_factor=scale, mode="bilinear", align_corners=False)

        return flow_out, mask_out
