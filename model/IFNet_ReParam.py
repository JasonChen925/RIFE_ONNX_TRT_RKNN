import torch
import torch.nn as nn
import torch.nn.functional as F
from warplayer import warp #生成onnx用,计算参数量用
# from model.warplayer import warp   #插帧用 训练用
from refine import * #生成onnx用
# from model.refine import *#插帧用 训练用


# --------- Re-parameterization Block ---------
class RepBlock(nn.Module):
    def __init__(self, channels):
        super(RepBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.prelu1 = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.prelu2 = nn.PReLU(channels)
        self.deploy = False

    def forward(self, x):
        if self.deploy:
            return self.prelu2(self.conv_reparam(x))
        else:
            x = self.prelu1(self.conv1(x))
            x = self.prelu2(self.conv2(x))
            return x

    def reparameterize(self):
        weight1 = self.conv1.weight
        bias1 = self.conv1.bias
        weight2 = self.conv2.weight
        bias2 = self.conv2.bias

        weight1 = weight1.view(self.conv1.out_channels, -1)
        weight2 = weight2.view(self.conv2.out_channels, -1)

        fused_weight = torch.matmul(weight2, weight1).view(self.conv2.out_channels, self.conv1.in_channels, 3, 3)
        fused_bias = torch.matmul(weight2, bias1) + bias2

        self.conv_reparam = nn.Conv2d(self.conv1.in_channels, self.conv2.out_channels, 3, 1, 1, bias=True)
        self.conv_reparam.weight.data = fused_weight
        self.conv_reparam.bias.data = fused_bias

        # 删除原来的两个卷积
        del self.conv1
        del self.conv2
        self.deploy = True


# --------- Basic conv/deconv block ---------
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PReLU(out_planes)
    )


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


# --------- IFBlock with Re-parameterization support ---------
class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            RepBlock(c),
            RepBlock(c),
            RepBlock(c),
            RepBlock(c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale=1):
        scale = float(scale)

        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)

        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask


# --------- IFNet ---------
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13 + 4, c=150)
        self.block2 = IFBlock(13 + 4, c=90)
        self.block_tea = IFBlock(16 + 4, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow is not None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)

        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                            scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None

        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])

        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill

from thop import profile
import torch

device = torch.device('cuda')
model = IFNet().to(device)
input = torch.randn(1, 6, 256, 256).to(device)  # 根据你的输入分辨率设置
flops, params = profile(model, inputs=(input,), verbose=False)

print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"参数量: {params / 1e6:.2f} M")