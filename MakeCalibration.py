import os
import cv2
import torch
import numpy as np
from glob import glob

# 设置数据集路径（多个子文件夹）
dataset_root = r"/home/jason/RIFE_ONNX_TRT_RKNN/vimeo_triplet/sequences/00001"
calibration_data = []

# 获取所有子文件夹（00001, 00002, ..., 10000）
subdirs = sorted(glob(os.path.join(dataset_root, "*")))[:64]  # 取前 64 组校准数据

for subdir in subdirs:
    img1_path = os.path.join(subdir, "im1.png")  # 第一帧
    img2_path = os.path.join(subdir, "im2.png")  # 中间帧（插值目标，不用于输入）
    img3_path = os.path.join(subdir, "im3.png")  # 第三帧

    # 读取图像
    img1 = cv2.imread(img1_path)
    img3 = cv2.imread(img3_path)

    if img1 is None or img3 is None:
        print(f"❌ 跳过无效文件: {img1_path} 或 {img3_path}")
        continue  # 遇到空文件，跳过

    # BGR 转 RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

    # 调整大小为 256x256（可修改）
    img1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
    img3 = cv2.resize(img3, (256, 256), interpolation=cv2.INTER_AREA)

    # 转换为 float32 并归一化到 [0,1]
    img1 = torch.tensor(img1, dtype=torch.float32) / 255.0
    img3 = torch.tensor(img3, dtype=torch.float32) / 255.0

    # HWC 转 CHW
    img1 = img1.permute(2, 0, 1)  # (H, W, C) → (C, H, W)
    img3 = img3.permute(2, 0, 1)  # (H, W, C) → (C, H, W)

    # 拼接成 (6, 256, 256) 作为输入
    img_pair = torch.cat((img1, img3), dim=0)  # (6, 256, 256)
    calibration_data.append(img_pair)

# 转换为 PyTorch 张量，形状 (N, 6, 256, 256)
calibration_data = torch.stack(calibration_data)

# 保存为 `.pt`
torch.save(calibration_data, "calibration_data.pt")
print(f"✅ 校准数据已保存！形状: {calibration_data.shape}, Dtype: {calibration_data.dtype}")
