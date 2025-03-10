import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from warplayer import *
from torch.ao.quantization import get_default_qconfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),        
        nn.PReLU(out_planes)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential( #降低特征分辨率,增加通道数,最终特征大小缩小为1/4
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock0 = nn.Sequential(  #convblock0-convblock3 包含两层卷积,输出特征维度保持不变,通过残差连接强化特征表示
            conv(c, c),
            conv(c, c)
        )
        self.convblock1 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock2 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock3 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.conv1 = nn.Sequential(  #光流估计分支
            #反卷积层(上采样)生成4通道光流,分别对应两个方向的光流(水平垂直各两通道)
            nn.ConvTranspose2d(c, c//2, 4, 2, 1),
            nn.PReLU(c//2),
            nn.ConvTranspose2d(c//2, 4, 4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            #反卷积层(上采样)生成1通道掩码,用于图像融合时加权使用
            nn.ConvTranspose2d(c, c//2, 4, 2, 1),
            nn.PReLU(c//2),
            nn.ConvTranspose2d(c//2, 1, 4, 2, 1),
        )

    def forward(self, x, flow, scale=1):  # x为输入的特征,包含两帧图像 ,flow:当前的光流信息,scale:缩放因子,用于调整分辨率
        #输入特征和光流通过双线性插值,调整到当前尺度
        #降低输入分辨率以适应特定的特征提取需求
        #光流也被重新缩放以保持一致性
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 1. / scale
        #初始特征提取,拼接输入特征和光流，通过 conv0 进行降采样特征提取
        feat = self.conv0(torch.cat((x, flow), 1))
        # 通过4个残差块,逐步提取更高层次的特征
        feat = self.convblock0(feat) + feat  #残差连接
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        #光流估计,提取到的高层次特征通过conv1解码为光流场
        flow = self.conv1(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * scale
        #掩码估计,同样的高层次特征通过conv2解码为掩码
        mask = self.conv2(feat)
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        return flow, mask
        
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()   #等价于直接调用父类的方法
        #每一个IFBlock是一个残差卷积块，负责多尺度光流的特征提取和掩码生成
        self.block0 = IFBlock(7+4, c=90)  #  7+4,c=90对应的是IFBlcok的__init__中的in_planes,c
        self.block1 = IFBlock(7+4, c=90)
        self.block2 = IFBlock(7+4, c=90)
        self.block_tea = IFBlock(10+4, c=90)  #教师网络，可能运用于训练过程中的知识蒸馏和指导学生网络
        # self.contextnet = Contextnet()
        # self.unet = Unet()

    def forward(self, x):      #输入x=[1,6,1440,2560]
        scale_list = [4, 2, 1]
        channel = x.shape[1] // 2   ## channel = 6//2=3
        img0 = x[:, :channel]    #img0:(1,3,1440,2560)
        img1 = x[:, channel:]    #img1:(1,3,1440,2560)

        flow_list = []
        merged = []
        mask_list = []

        warped_img0 = img0
        warped_img1 = img1
        flow = (x[:, :4]).detach() * 0  #flow:(1,4,1440,2560)
        mask = (x[:, :1]).detach() * 0  #mask:(1,1,1440,2560)
        loss_cons = 0
        block = [self.block0, self.block1, self.block2]
        # 三层循环计算
        for i in range(3):  #循环使用3个IFBlock
            #f0,m0使用正向光流，f1,m1使用反向光流
            f0, m0 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], mask), 1), flow, scale=scale_list[i])
            f1, m1 = block[i](torch.cat((warped_img1[:, :3], warped_img0[:, :3], -mask), 1), torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale=scale_list[i])
            # 更新flow和mask
            flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
            mask = mask + (m0 + (-m1)) / 2

            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))

        for i in range(3):#掩码加权融合
            mask_list[i] = torch.sigmoid(mask_list[i])
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            # merged[i] = torch.clamp(merged[i] + res, 0, 1)        
        return flow_list[0],flow_list[1],flow_list[2], mask_list[2], merged[0],merged[1],merged[2]



##################### 生成onnx模型文件#################################
# device = torch.device("cuda")
# dummy_input = torch.randn(1, 6, 1440,2560).to(device)  # 假设输入为 1440,2560 的图像
# model = IFNet().to(device)
# with torch.no_grad():
#     torch.onnx.export(model,
#                       dummy_input,
#                       "IFNet_fp32.onnx",
#                       opset_version = 16,
#                       input_names=['imgs'],
#                       output_names=['flow_list_0','flow_list_1','flow_list_2','mask_list_2','merged_0','merged_1','merged_2'])


###########################pytorch静态量化部分######################33
class QuantIFNet(nn.Module):
    def __init__(self):
        super(QuantIFNet, self).__init__()
        self.quant = torch.quantization.QuantStub()  # 量化输入
        self.model = IFNet()  # 原始光流插帧模型
        self.dequant = torch.quantization.DeQuantStub()  # 反量化输出

    def forward(self, x):
        x = self.quant(x)  # 量化输入
        x = self.model(x)  # 插帧处理
        x = self.dequant(x)  # 反量化输出
        return x

######## 量化设置（跳过 ConvTranspose2d）
# def apply_qconfig(module):
#     if isinstance(module, torch.nn.ConvTranspose2d):
#         module.qconfig = None  # 跳过量化
#     else:
#         module.qconfig = get_default_qconfig("fbgemm")

###设置qconfig并插入Observer
# torch.backends.quantized.engine = "fbgemm"##适用于x86 CPU
# #创建量化模型
# quantized_model = QuantIFNet().to("cpu")  #这里为什么用CPU
# quantized_model.eval()
# ##
# for name, module in quantized_model.named_modules():
#     if isinstance(module, torch.nn.ConvTranspose2d):
#         module.qconfig = None



# #插入Observer
# quantized_model = torch.quantization.prepare(quantized_model)
# # 读取校准数据
# calibration_data = torch.load('/home/jason/RIFE_ONNX_TRT_RKNN/ECCV2022-RIFE/calibration_data.pt')
#
# # 运行校准
# with torch.no_grad():
#     for i in range(len(calibration_data)-1):
#         quantized_model(calibration_data[i].unsqueeze(0).to("cpu"))
#
# #PyTorch 用 INT8 量化 CNN 权重,Observer 被移除,模型更小，推理更快
# quantized_model = torch.quantization.convert(quantized_model)
#
# # 导出量化后的ONNX
# dummy_input = torch.randn(1, 6, 256, 256).to("cpu")
#
# torch.onnx.export(quantized_model,
#                   dummy_input,
#                   "IFNet_quantized_torch_int8.onnx",
#                   opset_version=16,
#                   input_names=['imgs'],
#                   output_names=['output'])
# #########################以上为Pytorch静态量化部分###########################
#################### 生成onnx模型文件#################################
device = torch.device("cuda")
dummy_input = torch.randn(1, 6, 256,256).to(device)  # 假设输入为 1440,2560 的图像
model = IFNet().to(device)
with torch.no_grad():
    torch.onnx.export(model,
                      dummy_input,
                      "IFNet_256x256_fp32.onnx",
                      opset_version = 16,
                      input_names=['imgs'],
                      output_names=['flow_list_0','flow_list_1','flow_list_2','mask_list_2','merged_0','merged_1','merged_2'])
