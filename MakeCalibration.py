import os
import cv2
import torch
import numpy as np
from glob import glob

# 设置数据集路径
dataset_path = "/vimeo_triplet/sequences/00001/"
calibration_data = []

# 获取所有子文件夹（0001, 0002, ..., 1000）
subdirs = sorted(glob(os.path.join(dataset_path, "*")))[:32]  # 取前 32 组

for subdir in subdirs:
    img1_path = os.path.join(subdir, "im1.png")  # 第一帧
    img3_path = os.path.join(subdir, "im3.png")  # 第二帧

    # 读取图像并转换为张量
    img1 = cv2.imread(img1_path)
    img3 = cv2.imread(img3_path)

    # 调整大小为 256x256（可根据模型需要修改）
    img1 = cv2.resize(img1, (256, 256))
    img3 = cv2.resize(img3, (256, 256))

    # BGR 转 RGB，并归一化到 [0,1]
    img1 = torch.tensor(img1).permute(2, 0, 1) / 255.0
    img3 = torch.tensor(img3).permute(2, 0, 1) / 255.0

    # 拼接成 (6, 256, 256) 格式
    img_pair = torch.cat((img1, img3), dim=0)  # 拼接成 6 通道
    calibration_data.append(img_pair)

# 转换为 PyTorch 张量，形状 (32, 6, 256, 256)
calibration_data = torch.stack(calibration_data)

# 保存为 `.pt` 以便后续加载
torch.save(calibration_data, "calibration_data.pt")

print(f"校准数据已保存！形状: {calibration_data.shape}")  # (32, 6, 256, 256)
